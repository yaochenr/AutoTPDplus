"""Safe post-processing for PMC supplementary archives and legacy documents."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Optional


GIB = 1024**3
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_BYTES = 50 * GIB
MAX_COMPRESSION_RATIO = 10_000
SUPPLEMENT_EXTENSIONS = {
    ".pdf",
    ".xlsx",
    ".xls",
    ".docx",
    ".doc",
    ".csv",
    ".tsv",
    ".txt",
    ".md",
    ".zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_member_unsafe_reason(info: zipfile.ZipInfo) -> Optional[str]:
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
        "zip_path": str(zip_path),
        "archive_sha256": sha256_file(zip_path),
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
        elif len(infos) > MAX_ARCHIVE_MEMBERS:
            result["scan_status"] = "too_many_members"
        elif int(result["uncompressed_bytes"]) > MAX_ARCHIVE_BYTES:
            result["scan_status"] = "archive_too_large"
        elif float(result["max_member_ratio"]) > MAX_COMPRESSION_RATIO:
            result["scan_status"] = "suspicious_compression_ratio"
        else:
            result["scan_status"] = "safe"
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        result["scan_status"] = "invalid_zip"
        result["error"] = str(exc)
    return result


def _archive_destination(supplementary_dir: Path, zip_path: Path) -> Path:
    if zip_path.parent == supplementary_dir:
        return supplementary_dir / "extracted" / zip_path.stem
    return zip_path.with_suffix("")


def extract_zip(
    record: dict[str, Any],
    supplementary_dir: Path,
    *,
    archive_depth: int = 0,
) -> dict[str, Any]:
    """Extract a pre-scanned ZIP without deleting or replacing the archive."""
    record = dict(record)
    record["archive_depth"] = archive_depth
    if record["scan_status"] != "safe":
        record["extraction_status"] = "skipped_unsafe"
        return record

    zip_path = Path(str(record["zip_path"]))
    destination = _archive_destination(supplementary_dir, zip_path)
    manifest_path = destination / "extraction_manifest.json"
    source_archive = zip_path.relative_to(supplementary_dir).as_posix()
    archive_sha256 = str(record["archive_sha256"])

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") == "completed"
                and manifest.get("archive_sha256") == archive_sha256
            ):
                record.update({
                    "extraction_status": "already_extracted",
                    "extraction_dir": str(destination),
                    "source_archive": source_archive,
                    "files": manifest.get("files", []),
                })
                return record
        except (json.JSONDecodeError, OSError):
            pass

    if destination.exists():
        record["extraction_status"] = "destination_conflict"
        record["error"] = f"existing directory lacks a matching manifest: {destination}"
        return record

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    files: list[dict[str, Any]] = []
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
                final_output = destination.joinpath(*member.parts)
                files.append({
                    "source_archive": source_archive,
                    "archive_member": info.filename,
                    "relative_path": final_output.relative_to(supplementary_dir).as_posix(),
                    "size_bytes": info.file_size,
                    "sha256": sha256_file(output),
                })

        manifest = {
            "status": "completed",
            "source_archive": source_archive,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": zip_path.stat().st_size,
            "archive_depth": archive_depth,
            "extracted_files": len(files),
            "extracted_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (temporary / "extraction_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        os.replace(temporary, destination)
        record.update({
            "extraction_status": "extracted",
            "extraction_dir": str(destination),
            "source_archive": source_archive,
            "files": files,
        })
    except (OSError, ValueError, zipfile.BadZipFile, NotImplementedError) as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        record["extraction_status"] = "failed"
        record["error"] = str(exc)
    return record


def extract_zip_archives(
    supplementary_dir: Path,
    *,
    max_archive_depth: int = 2,
) -> list[dict[str, Any]]:
    """Extract top-level and nested ZIP SI up to a bounded archive depth."""
    supplementary_dir = Path(supplementary_dir)
    queue = [
        (path, 0)
        for path in sorted(supplementary_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".zip"
    ]
    visited: set[Path] = set()
    results: list[dict[str, Any]] = []
    while queue:
        zip_path, depth = queue.pop(0)
        resolved = zip_path.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        result = extract_zip(
            scan_zip(zip_path), supplementary_dir, archive_depth=depth
        )
        results.append(result)
        if depth >= max_archive_depth:
            continue
        extraction_dir = result.get("extraction_dir")
        if result.get("extraction_status") in {"extracted", "already_extracted"} and extraction_dir:
            queue.extend(
                (path, depth + 1)
                for path in sorted(Path(str(extraction_dir)).rglob("*"))
                if path.is_file() and path.suffix.lower() == ".zip"
            )
    return results


def convert_legacy_docs(
    supplementary_dir: Path,
    *,
    converter: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Convert legacy DOC files to derived DOCX files while keeping originals."""
    supplementary_dir = Path(supplementary_dir)
    if converter is None:
        converter = shutil.which("libreoffice") or shutil.which("soffice") or ""
    normalized_root = supplementary_dir / "normalized"
    records: list[dict[str, Any]] = []
    candidates = [
        path for path in supplementary_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".doc"
        and normalized_root not in path.parents
    ]
    for source in sorted(candidates):
        relative_source = source.relative_to(supplementary_dir)
        target = normalized_root / relative_source.with_suffix(".docx")
        source_sha256 = sha256_file(source)
        record: dict[str, Any] = {
            "source_path": relative_source.as_posix(),
            "source_sha256": source_sha256,
            "source_size_bytes": source.stat().st_size,
            "derived_path": target.relative_to(supplementary_dir).as_posix(),
        }
        if target.is_file():
            record.update({
                "status": "already_converted",
                "derived_sha256": sha256_file(target),
            })
        elif not converter:
            record["status"] = "converter_unavailable"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                completed = subprocess.run(
                    [converter, "--headless", "--convert-to", "docx", "--outdir", str(target.parent), str(source)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                generated = target.parent / f"{source.stem}.docx"
                if completed.returncode == 0 and generated.is_file():
                    if generated != target:
                        os.replace(generated, target)
                    record.update({
                        "status": "converted",
                        "derived_sha256": sha256_file(target),
                    })
                else:
                    record.update({
                        "status": "conversion_failed",
                        "error": (completed.stderr or completed.stdout)[-1000:],
                    })
            except (OSError, subprocess.SubprocessError) as exc:
                record.update({"status": "conversion_failed", "error": str(exc)})
        records.append(record)

    if records:
        normalized_root.mkdir(parents=True, exist_ok=True)
        (normalized_root / "document_normalization_manifest.json").write_text(
            json.dumps({"documents": records}, indent=2), encoding="utf-8"
        )
    return records
