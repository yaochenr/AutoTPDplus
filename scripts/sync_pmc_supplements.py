#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import stat
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlparse
from xml.etree import ElementTree

import requests

S3_URL = "https://pmc-oa-opendata.s3.amazonaws.com/"
USER_AGENT = "TPD-curator-supplement-sync/1.0 (mailto:research@example.com)"
DEFAULT_EXTENSIONS = {
    ".pdf",
    ".xlsx",
    ".xls",
    ".docx",
    ".csv",
    ".tsv",
    ".txt",
    ".md",
    ".zip",
    ".doc",
}
SUPPLEMENT_TAGS = {"supplementary-material", "supp-data"}
METADATA_FILENAME = "supplementary_metadata.json"
GIB = 1024**3


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalize_extensions(values: list[str]) -> set[str]:
    return {f".{value.lower().lstrip('.')}" for value in values}


def referenced_supplement_basenames(xml_path: Path) -> set[str]:
    if not xml_path.is_file():
        return set()
    try:
        root = ElementTree.parse(xml_path).getroot()
    except (ElementTree.ParseError, OSError):
        return set()

    names: set[str] = set()
    for element in root.iter():
        if local_name(element.tag) not in SUPPLEMENT_TAGS:
            continue
        for descendant in element.iter():
            for attribute, value in descendant.attrib.items():
                if local_name(attribute) != "href" or not value:
                    continue
                basename = Path(unquote(urlparse(value).path)).name
                if basename:
                    names.add(basename.casefold())
    return names


def is_supplement_filename(filename: str, referenced: set[str]) -> bool:
    lower = filename.casefold()
    if lower in referenced:
        return True
    return bool(re.search(r"-s\d+\.", lower)) or any(
        marker in lower
        for marker in ("supp", "supplement", "esm", "moesm", "si_", "mmc")
    )


def request_with_retry(
    url: str,
    *,
    params: dict[str, str] | None = None,
    stream: bool = False,
    timeout: int = 30,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                stream=stream,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last_error)) from last_error


def list_latest_s3_objects(pmcid: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        response = request_with_retry(
            S3_URL,
            params={"list-type": "2", "prefix": pmcid},
        )
        root = ElementTree.fromstring(response.content)
    except (RuntimeError, ElementTree.ParseError) as exc:
        return f"manifest_error: {exc}", []

    objects: list[dict[str, Any]] = []
    for content in root.iter():
        if local_name(content.tag) != "Contents":
            continue
        fields = {local_name(child.tag): child.text or "" for child in content}
        if fields.get("Key"):
            objects.append({
                "key": fields["Key"],
                "size_bytes": int(fields.get("Size") or 0),
            })

    version_pattern = re.compile(rf"^{re.escape(pmcid)}\.(\d+)/")
    versions = {
        int(match.group(1))
        for item in objects
        if (match := version_pattern.match(str(item["key"])))
    }
    if not versions:
        return "no_oa_package", []
    prefix = f"{pmcid}.{max(versions)}/"
    return "ok", [item for item in objects if str(item["key"]).startswith(prefix)]


def local_supplement_files(supplementary_dir: Path) -> dict[str, Path]:
    if not supplementary_dir.is_dir():
        return {}
    return {
        path.name.casefold(): path
        for path in supplementary_dir.iterdir()
        if path.is_file()
        and path.name not in {METADATA_FILENAME, ".DS_Store"}
        and not path.name.endswith(".part")
    }


def audit_paper(paper_dir: Path) -> dict[str, Any]:
    pmcid = paper_dir.name
    referenced = referenced_supplement_basenames(paper_dir / "full_text.xml")
    manifest_status, objects = list_latest_s3_objects(pmcid)
    remote: list[dict[str, Any]] = []
    if manifest_status == "ok":
        for item in objects:
            filename = Path(str(item["key"])).name
            if filename and is_supplement_filename(filename, referenced):
                remote.append({
                    **item,
                    "filename": filename,
                    "extension": Path(filename.casefold()).suffix,
                })

    supplementary_dir = paper_dir / "supplementary"
    local = local_supplement_files(supplementary_dir)
    missing = [
        item for item in remote if str(item["filename"]).casefold() not in local
    ]
    return {
        "pmcid": pmcid,
        "paper_dir": str(paper_dir),
        "supplementary_dir": str(supplementary_dir),
        "manifest_status": manifest_status,
        "local_file_count": len(local),
        "remote_file_count": len(remote),
        "missing_file_count": len(missing),
        "remote_files": remote,
        "missing_files": missing,
        "xml_referenced_filenames": sorted(referenced),
    }


def download_object(article: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    supplementary_dir = Path(str(article["supplementary_dir"]))
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    target = supplementary_dir / str(item["filename"])
    partial = target.with_name(target.name + ".part")
    expected_size = int(item.get("size_bytes") or 0)

    if target.is_file() and target.stat().st_size > 0:
        return {
            "pmcid": article["pmcid"],
            "filename": target.name,
            "status": "already_present",
            "size_bytes": target.stat().st_size,
            "path": str(target),
        }

    try:
        with request_with_retry(
            S3_URL + quote(str(item["key"]), safe="/"),
            stream=True,
            timeout=60,
        ) as response:
            http_size = int(response.headers.get("Content-Length", "0") or 0)
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        actual_size = partial.stat().st_size
        required_size = expected_size or http_size
        if actual_size == 0:
            raise OSError("downloaded file is empty")
        if required_size and actual_size != required_size:
            raise OSError(
                f"size mismatch: expected {required_size}, received {actual_size}"
            )
        os.replace(partial, target)
        return {
            "pmcid": article["pmcid"],
            "filename": target.name,
            "status": "downloaded",
            "size_bytes": actual_size,
            "path": str(target),
        }
    except (RuntimeError, OSError) as exc:
        partial.unlink(missing_ok=True)
        return {
            "pmcid": article["pmcid"],
            "filename": target.name,
            "status": "download_failed",
            "size_bytes": 0,
            "path": str(target),
            "error": str(exc),
        }


def update_supplementary_metadata(paper_dir: Path) -> None:
    supplementary_dir = paper_dir / "supplementary"
    if not supplementary_dir.is_dir():
        return
    metadata_path = supplementary_dir / METADATA_FILENAME
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}

    metadata.setdefault("pmcid", paper_dir.name)
    metadata.setdefault("footnote_references", [])
    metadata.setdefault("external_links", [])
    existing_entries = {
        entry.get("filename"): entry
        for entry in metadata.get("files", [])
        if isinstance(entry, dict) and entry.get("filename")
    }
    files = sorted(local_supplement_files(supplementary_dir).values())
    entries: list[dict[str, Any]] = []
    for path in files:
        entry = existing_entries.get(path.name, {})
        entry.update({
            "id": entry.get("id") or path.stem,
            "filename": path.name,
            "caption": entry.get("caption") or "Supplementary file",
            "file_type": path.suffix.lower().lstrip(".") or "unknown",
            "size_bytes": path.stat().st_size,
            "local_path": str(path),
            "content_summary": entry.get("content_summary")
            or f"{path.suffix.upper().lstrip('.')} supplementary file",
        })
        entries.append(entry)

    metadata.update({
        "has_supplements": bool(files),
        "files": entries,
        "download_status": "completed",
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def zip_member_unsafe_reason(info: zipfile.ZipInfo) -> str | None:
    normalized = info.filename.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or member_path.is_absolute():
        return "absolute_or_empty_path"
    if any(part == ".." for part in member_path.parts):
        return "path_traversal"
    if stat.S_ISLNK(info.external_attr >> 16):
        return "symbolic_link"
    if info.flag_bits & 0x1:
        return "encrypted_member"
    return None


def scan_zip(zip_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pmcid": zip_path.parent.parent.name,
        "filename": zip_path.name,
        "zip_path": str(zip_path),
        "compressed_bytes": zip_path.stat().st_size,
        "scan_status": "pending",
        "member_count": 0,
        "file_count": 0,
        "uncompressed_bytes": 0,
        "max_member_ratio": 0.0,
    }
    try:
        with zipfile.ZipFile(zip_path) as archive:
            infos = archive.infolist()
        unsafe = [
            {"member": info.filename, "reason": reason}
            for info in infos
            if (reason := zip_member_unsafe_reason(info))
        ]
        result.update({
            "member_count": len(infos),
            "file_count": sum(not info.is_dir() for info in infos),
            "uncompressed_bytes": sum(
                info.file_size for info in infos if not info.is_dir()
            ),
            "max_member_ratio": round(
                max(
                    (
                        info.file_size / max(info.compress_size, 1)
                        for info in infos
                        if not info.is_dir() and info.file_size
                    ),
                    default=0.0,
                ),
                2,
            ),
            "unsafe_members": unsafe,
        })
        if unsafe:
            result["scan_status"] = "unsafe"
        elif len(infos) > 100000:
            result["scan_status"] = "too_many_members"
        elif int(result["uncompressed_bytes"]) > 50 * GIB:
            result["scan_status"] = "archive_too_large"
        elif float(result["max_member_ratio"]) > 10000:
            result["scan_status"] = "suspicious_compression_ratio"
        else:
            result["scan_status"] = "safe"
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        result["scan_status"] = "invalid_zip"
        result["error"] = str(exc)
    return result


def extract_zip(record: dict[str, Any]) -> dict[str, Any]:
    if record["scan_status"] != "safe":
        record["extraction_status"] = "skipped_unsafe"
        return record

    zip_path = Path(str(record["zip_path"]))
    extracted_root = zip_path.parent / "extracted"
    destination = extracted_root / zip_path.stem
    manifest_path = destination / "extraction_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") == "completed"
                and manifest.get("zip_size_bytes") == zip_path.stat().st_size
            ):
                record.update({
                    "extraction_status": "already_extracted",
                    "extraction_dir": str(destination),
                    "extracted_files": manifest.get("extracted_files", 0),
                    "extracted_bytes": manifest.get("extracted_bytes", 0),
                })
                return record
        except (json.JSONDecodeError, OSError):
            pass
    if destination.exists():
        record["extraction_status"] = "destination_conflict"
        record["error"] = f"existing directory lacks a matching manifest: {destination}"
        return record

    temporary = extracted_root / f".{zip_path.stem}.extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    extracted_files = 0
    extracted_bytes = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if reason := zip_member_unsafe_reason(info):
                    raise ValueError(f"unsafe member {info.filename}: {reason}")
                member = PurePosixPath(info.filename.replace("\\", "/"))
                output = temporary.joinpath(*member.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                partial = output.with_name(output.name + ".part")
                with archive.open(info) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if partial.stat().st_size != info.file_size:
                    raise OSError(
                        f"size mismatch for {info.filename}: "
                        f"expected {info.file_size}, received {partial.stat().st_size}"
                    )
                os.replace(partial, output)
                extracted_files += 1
                extracted_bytes += info.file_size
        os.replace(temporary, destination)
        manifest_path.write_text(
            json.dumps({
                "status": "completed",
                "zip_path": str(zip_path),
                "zip_size_bytes": zip_path.stat().st_size,
                "extracted_files": extracted_files,
                "extracted_bytes": extracted_bytes,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2),
            encoding="utf-8",
        )
        record.update({
            "extraction_status": "extracted",
            "extraction_dir": str(destination),
            "extracted_files": extracted_files,
            "extracted_bytes": extracted_bytes,
        })
    except (OSError, ValueError, zipfile.BadZipFile, NotImplementedError) as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        record["extraction_status"] = "failed"
        record["error"] = str(exc)
    return record


def count_status(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_reports(
    report_dir: Path,
    report_stem: str,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_stem}.json"
    csv_path = report_dir / f"{report_stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = [
        "pmcid",
        "manifest_status",
        "local_file_count",
        "remote_file_count",
        "missing_file_count",
        "selected_missing_count",
        "remaining_selected_missing_count",
        "missing_files",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for article in payload["articles"]:
            row = dict(article)
            row["missing_files"] = ";".join(
                str(item["filename"]) for item in article["missing_files"]
            )
            writer.writerow(row)
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize PMC OA supplementary materials and safely extract ZIPs."
    )
    parser.add_argument("--papers-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_EXTENSIONS),
        help="Extensions to download (default: common documents, tables, ZIP and DOC).",
    )
    parser.add_argument("--extract-zips", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--report-stem",
        default=f"supplementary_sync_{time.strftime('%Y-%m-%d')}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected_extensions = normalize_extensions(args.extensions)
    paper_dirs = sorted(
        path for path in args.papers_dir.glob("PMC*") if path.is_dir()
    )

    articles: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_paper, path): path for path in paper_dirs}
        for future in as_completed(futures):
            articles.append(future.result())
    articles.sort(key=lambda item: str(item["pmcid"]))

    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for article in articles:
        selected_missing = [
            item
            for item in article["missing_files"]
            if str(item["extension"]) in selected_extensions
        ]
        article["selected_missing_count"] = len(selected_missing)
        if not args.audit_only:
            jobs.extend((article, item) for item in selected_missing)

    downloads: list[dict[str, Any]] = []
    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_object, article, item): (article, item)
                for article, item in jobs
            }
            for future in as_completed(futures):
                downloads.append(future.result())
        downloads.sort(key=lambda item: (str(item["pmcid"]), str(item["filename"])))
        changed_pmcids = {
            str(item["pmcid"])
            for item in downloads
            if item["status"] in {"downloaded", "already_present"}
        }
        for pmcid in changed_pmcids:
            update_supplementary_metadata(args.papers_dir / pmcid)

    for article in articles:
        local_after = local_supplement_files(Path(str(article["supplementary_dir"])))
        article["remaining_selected_missing_count"] = sum(
            str(item["extension"]) in selected_extensions
            and str(item["filename"]).casefold() not in local_after
            for item in article["remote_files"]
        )

    zip_extractions: list[dict[str, Any]] = []
    if args.extract_zips and not args.audit_only:
        zip_paths = sorted({
            path
            for article in articles
            for path in local_supplement_files(
                Path(str(article["supplementary_dir"]))
            ).values()
            if path.suffix.casefold() == ".zip"
        })
        zip_extractions = [scan_zip(path) for path in zip_paths]
        bytes_needed = sum(
            int(record["uncompressed_bytes"])
            for record in zip_extractions
            if record["scan_status"] == "safe"
            and not (Path(str(record["zip_path"])).parent / "extracted"
                     / Path(str(record["zip_path"])).stem
                     / "extraction_manifest.json").is_file()
        )
        free_bytes = shutil.disk_usage(args.papers_dir).free
        if free_bytes >= bytes_needed + 5 * GIB:
            zip_extractions = [extract_zip(record) for record in zip_extractions]
        else:
            for record in zip_extractions:
                if record["scan_status"] == "safe":
                    record["extraction_status"] = "skipped_insufficient_space"

    successful_manifests = sum(
        article["manifest_status"] == "ok" for article in articles
    )
    summary = {
        "papers_checked": len(articles),
        "manifest_checks_successful": successful_manifests,
        "manifest_check_errors": len(articles) - successful_manifests,
        "selected_extensions": sorted(selected_extensions),
        "audit_only": args.audit_only,
        "selected_missing_files_before": sum(
            int(article["selected_missing_count"]) for article in articles
        ),
        "selected_missing_files_after": sum(
            int(article["remaining_selected_missing_count"]) for article in articles
        ),
        "download_status_counts": count_status(downloads, "status"),
        "downloaded_bytes": sum(
            int(item.get("size_bytes", 0))
            for item in downloads
            if item["status"] == "downloaded"
        ),
        "zip_archives": len(zip_extractions),
        "zip_scan_status_counts": count_status(zip_extractions, "scan_status"),
        "zip_extraction_status_counts": count_status(
            zip_extractions, "extraction_status"
        ),
    }
    payload = {
        "summary": summary,
        "articles": articles,
        "downloads": downloads,
        "zip_extractions": zip_extractions,
    }
    json_path, csv_path = write_reports(
        args.report_dir, args.report_stem, payload
    )
    print(json.dumps(summary, indent=2))
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")

if __name__ == "__main__":
    main()
