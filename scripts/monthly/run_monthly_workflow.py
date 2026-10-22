#!/usr/bin/env python3




























from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DEFAULT_CONFIG = REPO / "tpd_curator" / "config" / "monthly_update_workflow.yaml"
MODALITIES = ("MG", "PROTAC")
PAPER_DIRS = {"MG": "MGs", "PROTAC": "PROTACs"}
STAGES = ["pmc", "extract", "post", "standardize", "merge", "qc"]
STD_SCRIPT = "scripts/data_standardization.py"
REQUIRED_STD_COLUMNS = [
    "DOI", "Compound_Name", "Degradation_Target", "Cell_Line", "DC50", "DC50_units", "Dmax",
    "Cell_Line_STD", "Degradation_Target_Gene", "Recruiter_Gene", "Assay_STD", "Compound_Key",
]
ACTIVITY_COLUMNS = ["DC50", "DC50_range_min", "DC50_range_max", "Dmax", "Dmax_range_min", "Dmax_range_max"]
RECORD_KEY = ["DOI", "Compound_Name", "Degradation_Target", "Recruiter", "Cell_Line", "Assay",
              "DC50", "DC50_units", "DC50_h", "Dmax", "Dmax_h"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = _load("build_autotpdplus", REPO / "scripts" / "build_autotpdplus.py")
mapping_agent = _load(
    "update_standardization_mappings",
    REPO / "scripts" / "monthly" / "update_standardization_mappings.py",
)


class StageError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def csv_rows(path: Path) -> int | None:
    return len(pd.read_csv(path, dtype=str, keep_default_na=False)) if path.exists() else None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()






class Context:
    def __init__(self, month: str, config: dict[str, Any], repo: Path = REPO, work_root: Path | None = None):
        self.month = month
        self.config = config
        self.repo = repo
        self.work_root = work_root
        self.status: dict[str, dict[str, Any]] = {}

    def path(self, value: str | Path) -> Path:
        value = Path(value)
        return value if value.is_absolute() else self.repo / value

    @property
    def updates(self) -> Path:
        return self.path(self.config["paths"]["updates_root"]) / self.month

    @property
    def work(self) -> Path:
        return (self.work_root or self.path(self.config["paths"]["work_root"])) / self.month

    @property
    def papers(self) -> Path:
        return self.path(self.config["pmc"]["papers_root"]) / self.month / "papers"

    @property
    def snapshot(self) -> Path:
        return self.work / "pre_merge"

    def release(self, modality: str) -> Path:
        return self.path(self.config["paths"]["release"][modality])

    @property
    def state(self) -> Path:
        return self.path(self.config["paths"]["state"])

    @property
    def retired(self) -> Path:
        return self.path(self.config["paths"]["retired_ids"])

    def log(self, name: str) -> Path:
        path = self.work / "logs" / f"{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def mark(self, stage: str, status: str, **detail: Any) -> None:
        self.status[stage] = {"status": status, "at": utc_now(), **detail}
        write_json(self.work / "status.json", self.status)


def run_logged(cmd: list[str], log_path: Path, cwd: Path = REPO, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}\n  log: {log_path}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {utc_now()} $ {printable}\n")
        log.flush()
        completed = subprocess.run(
            [str(part) for part in cmd], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
        )
    if completed.returncode != 0:
        raise StageError(f"command failed with exit code {completed.returncode}: {printable} (see {log_path})")


def python_env() -> dict[str, str]:
    return {"PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}






def stage_pmc(ctx: Context) -> None:
    cfg = ctx.config["pmc"]
    work = ctx.work / "pmc"
    rejected = sorted(ctx.path(ctx.config["paths"]["updates_root"]).glob("*/reviewer_rejected_dois.txt"))
    cmd = [
        "pixi", "run", "--manifest-path", Path(cfg["repo"]) / "pyproject.toml",
        "python", REPO / "scripts" / "monthly" / "pmc_stage.py",
        "--registry", cfg["registry"], "--papers_dir", ctx.papers, "--work_dir", work,
        "--run_label", ctx.month,
    ]
    if rejected:
        cmd += ["--rejected_dois", *rejected]
    if cfg.get("include_reviews"):
        cmd.append("--include_reviews")
    if not cfg.get("download_images"):
        cmd.append("--no_images")
    if cfg.get("max_results_per_query"):
        cmd += ["--max_results_per_query", str(cfg["max_results_per_query"])]
    run_logged(cmd, ctx.log("pmc"), cwd=Path(cfg["repo"]))

    out = ctx.updates / "pmc"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("pmc_stage_summary.json", "accepted_mg.json", "accepted_protac.json",
                 "article_types.json", "policy_decisions.csv", "classification.json"):
        if (work / name).exists():
            shutil.copy2(work / name, out / name)
    template = ctx.updates / "reviewer_rejected_dois.txt"
    if not template.exists():
        template.write_text(
            "# One DOI per line. Rows of these DOIs are removed from this month's merge by CI,\n"
            "# and the articles are marked rejected_by_reviewer in the registry next month.\n",
            encoding="utf-8",
        )
    summary = read_json(out / "pmc_stage_summary.json", {})
    ctx.mark("pmc", "done", accepted=summary.get("accepted_by_modality"))


def accepted_list(ctx: Context, modality: str) -> list[dict[str, Any]]:
    return read_json(ctx.updates / "pmc" / f"accepted_{modality.lower()}.json", []) or []


def extraction_files(results_dir: Path) -> list[Path]:

    suffixes = ("_extraction.json", "_supplementary_extraction.json", "_final_review.json")
    return [p for p in results_dir.rglob("*.json") if p.name.endswith(suffixes)]


def stage_extract(ctx: Context) -> None:
    cfg = ctx.config["extraction"]
    counts = {}
    for modality in MODALITIES:
        papers = accepted_list(ctx, modality)
        work = ctx.work / modality
        target = work / "data_processed.csv"
        if not papers:
            counts[modality] = 0
            continue
        paper_dir = ctx.papers / PAPER_DIRS[modality]
        if cfg.get("sync_supplements", True):
            run_logged([sys.executable, REPO / "scripts" / "sync_pmc_supplements.py",
                        "--papers-dir", paper_dir, "--report-dir", work / "supplements", "--extract-zips"],
                       ctx.log(f"extract_{modality}"), env=python_env())


        for name in ("output", "results"):
            shutil.rmtree(work / name, ignore_errors=True)
        runtime = yaml.safe_load(ctx.path(cfg["base_config"]).read_text(encoding="utf-8"))
        filter_file = ctx.updates / "pmc" / f"accepted_{modality.lower()}.json"
        runtime["data"].update({
            "json_path": str(paper_dir),
            "pdf_path": str(paper_dir),
            "output_dir": str(work / "output"),
            "paper_filter_file": str(filter_file),
            "paper_ids_file": str(filter_file),
        })
        runtime["results"]["results_dir"] = str(work / "results")
        runtime_path = work / "runtime_config.yaml"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True), encoding="utf-8")

        run_logged([sys.executable, REPO / "scripts" / "run_pipeline.py",
                    "--config_path", runtime_path,
                    "--input_prompt", ctx.path(cfg["prompts"][modality]),
                    *cfg["pipeline_flags"]],
                   ctx.log(f"extract_{modality}"), env=python_env())
        found = sorted((work / "results").glob("*/data_processed.csv"))
        if not found and not extraction_files(work / "results"):

            counts[modality] = 0
            continue
        if len(found) != 1:
            raise StageError(f"{modality}: expected one data_processed.csv under {work / 'results'}, found {len(found)}")
        shutil.copy2(found[0], target)
        counts[modality] = csv_rows(target)
    ctx.mark("extract", "done", rows=counts)


def stage_post(ctx: Context) -> None:
    counts = {}
    cache = ctx.config.get("post_processing", {}).get("cache_json")
    for modality in MODALITIES:
        source = ctx.work / modality / "data_processed.csv"
        if not accepted_list(ctx, modality) or not source.exists():
            continue
        target = ctx.updates / modality / "data_post_processed.csv"
        cmd = [sys.executable, REPO / "scripts" / "post_processing.py",
               "--input_csv", source, "--output_csv", target]
        if cache:
            cmd += ["--cache_json", ctx.path(cache)]
        run_logged(cmd, ctx.log(f"post_{modality}"), env=python_env())
        counts[modality] = csv_rows(target)
    ctx.mark("post", "done", rows=counts)


def standardize(ctx: Context, modality: str, output: Path, report: Path) -> None:
    run_logged([sys.executable, ctx.repo / STD_SCRIPT,
                "--input_csv", ctx.updates / modality / "data_post_processed.csv",
                "--output_csv", output, "--unmapped_report", report],
               ctx.log(f"standardize_{modality}"), cwd=ctx.repo, env=python_env())


def stage_standardize(ctx: Context, use_agent: bool = True) -> None:
    before = ctx.work / "data_standardization_before.py"
    if not before.exists():
        before.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ctx.repo / STD_SCRIPT, before)

    modalities = [m for m in MODALITIES if (ctx.updates / m / "data_post_processed.csv").exists()]
    reports = []
    for modality in modalities:
        report = ctx.work / modality / "unmapped.json"
        standardize(ctx, modality, ctx.work / modality / "std_pass1.csv", report)
        reports.append(report)

    merged = mapping_agent.merge_reports(reports)
    write_json(ctx.updates / "unmapped_standardization_values.json", {"items": merged})

    applied = 0
    agent_cfg = ctx.config.get("mapping_agent", {})
    if use_agent and not reports:

        write_json(ctx.updates / "mapping_changes.json",
                   {"run_label": ctx.month, "counts": {"applied": 0}, "changes": [], "key_lines": {}})
        (ctx.updates / "standardization_mapping_require_review.md").write_text(
            mapping_agent.render_review([], {}, ctx.month, Path(STD_SCRIPT).name), encoding="utf-8")
    elif use_agent:
        cmd = [sys.executable, REPO / "scripts" / "monthly" / "update_standardization_mappings.py",
               "--unmapped_report", *reports, "--standardization_script", ctx.repo / STD_SCRIPT,
               "--run_label", ctx.month,
               "--changes_json", ctx.updates / "mapping_changes.json",
               "--review_md", ctx.updates / "standardization_mapping_require_review.md"]
        if not agent_cfg.get("enabled", True) or not merged:
            cmd.append("--no_llm")
        if agent_cfg.get("model"):
            cmd += ["--model", agent_cfg["model"]]
        run_logged(cmd, ctx.log("mapping_agent"), env=python_env())
        applied = read_json(ctx.updates / "mapping_changes.json", {}).get("counts", {}).get("applied", 0)

    rows = {}
    for modality in modalities:
        final = ctx.updates / modality / "data_processed_std.csv"
        if applied:
            standardize(ctx, modality, final, ctx.work / modality / "unmapped_after.json")
        else:
            shutil.copy2(ctx.work / modality / "std_pass1.csv", final)
        rows[modality] = csv_rows(final)
    ctx.mark("standardize", "done", rows=rows, mappings_applied=applied)


def snapshot_release(ctx: Context, dest: Path) -> None:

    if (dest / "complete").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for modality in MODALITIES:
        shutil.copy2(ctx.release(modality), dest / f"{modality}.csv")
    shutil.copy2(ctx.state, dest / "state.json")
    if ctx.retired.exists():
        shutil.copy2(ctx.retired, dest / "retired_ids.csv")
    (dest / "complete").write_text(utc_now(), encoding="utf-8")


def snapshot_from_git(ctx: Context, ref: str, dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    files = {f"{m}.csv": ctx.config["paths"]["release"][m] for m in MODALITIES}
    files["state.json"] = ctx.config["paths"]["state"]
    files["retired_ids.csv"] = ctx.config["paths"]["retired_ids"]
    for name, rel in files.items():
        shown = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ctx.repo, capture_output=True)
        if shown.returncode == 0:
            (dest / name).write_bytes(shown.stdout)
        elif name != "retired_ids.csv":
            raise StageError(f"{rel} not found at {ref}")
    (dest / "complete").write_text(ref, encoding="utf-8")


def merge_from_snapshot(ctx: Context, snapshot: Path) -> dict[str, Any]:

    state = build.load_state(snapshot / "state.json")
    next_ids = state.get("next_id", {})
    retired_before = build.load_retired(snapshot / "retired_ids.csv")
    article_types = build.load_article_types(
        ctx.updates / "pmc" / "article_types.json" if (ctx.updates / "pmc" / "article_types.json").exists() else None
    )
    rejected = build.read_doi_list(ctx.updates / "reviewer_rejected_dois.txt")
    results = {}
    for modality in MODALITIES:
        if modality not in next_ids:
            raise StageError(f"no next_id for {modality} in {snapshot / 'state.json'}; run bootstrap first")
        batch_path = ctx.updates / modality / "data_processed_std.csv"
        batch = build.read_table(batch_path) if batch_path.exists() else pd.DataFrame(columns=["DOI"])
        results[modality] = build.incremental(
            build.read_table(snapshot / f"{modality}.csv"), batch,
            modality=modality, next_id=int(next_ids[modality]), run_label=ctx.month,
            retired_ids=set(retired_before[build.ID_COLUMN]),
            rejected_dois=rejected, article_types=article_types,
        )
    return {"state": state, "retired_before": retired_before, "results": results}


def stage_merge(ctx: Context, snapshot: Path) -> None:
    merged = merge_from_snapshot(ctx, snapshot)
    results = merged["results"]
    errors = {m: r.report["validation_errors"] for m, r in results.items() if r.report["validation_errors"]}
    write_json(ctx.updates / "merge_report.json", {m: r.report for m, r in results.items()})
    conflicts = pd.concat([r.conflicts.assign(modality=m) for m, r in results.items()], ignore_index=True)
    build.write_table(conflicts[["modality"] + build.CONFLICT_COLUMNS], ctx.updates / "merge_conflicts.csv")
    if errors:
        raise StageError(f"merge validation failed: {errors}")

    for modality, result in results.items():
        build.write_table(result.release, ctx.release(modality))
    state = merged["state"]
    for modality, result in results.items():
        state.setdefault("next_id", {})[modality] = result.next_id
    build.save_state(ctx.state, state)
    new_retired = [r.retired for r in results.values() if len(r.retired)]
    if new_retired:
        build.write_table(pd.concat([merged["retired_before"], *new_retired], ignore_index=True), ctx.retired)
    ctx.mark("merge", "done", appended={m: r.report["appended_rows"] for m, r in results.items()})






def _check(checks: list[dict[str, Any]], name: str, status: str, detail: Any = "") -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def mapping_diff(before_path: Path, after_path: Path) -> dict[str, dict[str, list[str]]]:
    before = mapping_agent.load_module(before_path, "std_qc_before")
    after = mapping_agent.load_module(after_path, "std_qc_after")
    diff = {}
    for name in mapping_agent.KIND_MAPPING.values():
        old, new = getattr(before, name), getattr(after, name)
        diff[name] = {
            "added": sorted(str(k) for k in new if k not in old),
            "removed": sorted(str(k) for k in old if k not in new),
            "changed": sorted(str(k) for k in old if k in new and old[k] != new[k]),
        }
    return diff


def stage_qc(ctx: Context, snapshot: Path, mapping_base: Path | None, reviewer_edits_allowed: bool) -> dict[str, Any]:
    normalize_doi = build.normalize_doi
    checks: list[dict[str, Any]] = []
    for modality in MODALITIES:
        std_path = ctx.updates / modality / "data_processed_std.csv"
        if not std_path.exists():
            continue
        std = build.read_table(std_path)
        accepted = {normalize_doi(p.get("doi")) for p in accepted_list(ctx, modality)}
        if accepted:
            stray = sorted(set(std["DOI"].map(normalize_doi)) - accepted)
            _check(checks, f"{modality}: rows come from {modality} articles", "fail" if stray else "pass", stray[:10])
        missing = [c for c in REQUIRED_STD_COLUMNS if c not in std.columns]
        activity = [c for c in ACTIVITY_COLUMNS if c in std.columns]
        if activity:

            empty = int((std[activity] == "").all(axis=1).sum())
            _check(checks, f"{modality}: DC50 or Dmax present", "fail" if empty else "pass", f"{empty} rows without either")
        _check(checks, f"{modality}: standardized schema", "fail" if missing else "pass", missing)

    changes = read_json(ctx.updates / "mapping_changes.json", {}) or {}
    applied = [c for c in changes.get("changes", []) if c["status"] == "applied"]
    if mapping_base and mapping_base.exists():
        diff = mapping_diff(mapping_base, ctx.repo / STD_SCRIPT)
        touched = {n: d["removed"] + d["changed"] for n, d in diff.items() if d["removed"] or d["changed"]}
        status = "warn" if reviewer_edits_allowed else "fail"
        _check(checks, "existing mappings not overwritten", status if touched else "pass", touched)
        documented = {(c["mapping_name"], str(c["value"])) for c in applied}
        undocumented = {n: [k for k in d["added"] if (n, k) not in documented] for n, d in diff.items()}
        undocumented = {n: keys for n, keys in undocumented.items() if keys}
        _check(checks, "every added mapping documented in the review file",
               ("warn" if reviewer_edits_allowed else "fail") if undocumented else "pass", undocumented)
    review = ctx.updates / "standardization_mapping_require_review.md"
    if applied:
        text = review.read_text(encoding="utf-8") if review.exists() else ""
        missing_md = [c["value"] for c in applied if f"`{mapping_agent._cell(c['value'])}`" not in text]
        _check(checks, "review Markdown lists every agent change", "fail" if missing_md else "pass", missing_md)

    report = read_json(ctx.updates / "merge_report.json")
    if report and (snapshot / "complete").exists():
        for modality, rep in report.items():
            ok = not rep["validation_errors"] and rep["rows_reconcile"]
            _check(checks, f"{modality}: merge validation and row reconciliation", "pass" if ok else "fail",
                   rep["validation_errors"] or f"{rep['previous_rows']} - {rep['retired_rows']} + {rep['appended_rows']} = {rep['release_rows']}")
            release = build.read_table(ctx.release(modality))
            _check(checks, f"{modality}: AutoTPD_ID unique", "pass" if release[build.ID_COLUMN].is_unique else "fail")
            previous = build.read_table(snapshot / f"{modality}.csv")
            merged = release.set_index(build.ID_COLUMN)
            carried = previous.set_index(build.ID_COLUMN)
            present = carried.index.isin(merged.index)
            retired_n = int((~present).sum())
            same = (merged.loc[carried.index[present], carried.columns] == carried[present]).all(axis=1)
            unchanged = bool(same.all()) and retired_n == rep["retired_rows"]
            _check(checks, f"{modality}: historical rows and IDs unchanged", "pass" if unchanged else "fail",
                   f"{int((~same).sum())} changed rows, {retired_n} retired")
            appended = release[~release[build.ID_COLUMN].isin(previous[build.ID_COLUMN])]
            key = [c for c in RECORD_KEY if c in appended.columns]
            dups = int(appended.duplicated(subset=key).sum()) if len(appended) else 0
            _check(checks, f"{modality}: no duplicate record keys in new rows", "warn" if dups else "pass", f"{dups} rows")
            n_conflicts = sum(rep["conflict_rows"].values())
            _check(checks, f"{modality}: merge conflicts", "warn" if n_conflicts else "pass", rep["conflict_rows"])
        rerun = merge_from_snapshot(ctx, snapshot)["results"]
        same_output = all(
            rerun[m].release.equals(build.read_table(ctx.release(m))) for m in MODALITIES
        )
        _check(checks, "merge is deterministic (re-run gives identical release)", "pass" if same_output else "fail")

    pmc = read_json(ctx.updates / "pmc" / "pmc_stage_summary.json", {}) or {}
    if pmc.get("download_failures") or pmc.get("screen_errors"):
        _check(checks, "PMC downloads and screening", "warn",
               {"download_failures": pmc.get("download_failures"), "screen_errors": pmc.get("screen_errors")})

    qc = {
        "checked_at": utc_now(),
        "failed": sum(c["status"] == "fail" for c in checks),
        "warnings": sum(c["status"] == "warn" for c in checks),
        "checks": checks,
    }
    write_json(ctx.updates / "qc_report.json", qc)
    return qc


def restore_release(ctx: Context, snapshot: Path) -> None:
    for modality in MODALITIES:
        shutil.copy2(snapshot / f"{modality}.csv", ctx.release(modality))
    shutil.copy2(snapshot / "state.json", ctx.state)
    if (snapshot / "retired_ids.csv").exists():
        shutil.copy2(snapshot / "retired_ids.csv", ctx.retired)
    elif ctx.retired.exists():
        ctx.retired.unlink()






def write_summary(ctx: Context, outcome: str, failure: str | None = None) -> Path:
    pmc = read_json(ctx.updates / "pmc" / "pmc_stage_summary.json", {}) or {}
    changes = read_json(ctx.updates / "mapping_changes.json", {}) or {}
    merge = read_json(ctx.updates / "merge_report.json", {}) or {}
    qc = read_json(ctx.updates / "qc_report.json", {}) or {}
    rows = {m: {
        "extracted": csv_rows(ctx.work / m / "data_processed.csv"),
        "post_processed": csv_rows(ctx.updates / m / "data_post_processed.csv"),
        "standardized": csv_rows(ctx.updates / m / "data_processed_std.csv"),
    } for m in MODALITIES}

    out = [f"# AutoTPD+ monthly update {ctx.month}", "", f"- Outcome: **{outcome}**", f"- Written: {utc_now()}"]
    if failure:
        out += [f"- Failure: `{failure}`"]
    out += ["", "| Stage | Status | Detail |", "|---|---|---|"]
    for stage in STAGES:
        info = ctx.status.get(stage, {"status": "not run"})
        detail = {k: v for k, v in info.items() if k not in {"status", "at"}}
        text = json.dumps(detail, ensure_ascii=False) if detail else ""
        out.append(f"| {stage} | {info['status']} | {text[:200] + ('…' if len(text) > 200 else '')} |")

    out += ["", "## Articles", ""]
    if pmc:
        out += [
            f"- New candidates screened: {pmc.get('new_candidates', 0)}",
            f"- Screening outcome: {json.dumps(pmc.get('screening_status_counts', {}))}",
            f"- Manual-review policies: {json.dumps(pmc.get('policy_counts', {}))}",
            f"- Accepted and downloaded: {json.dumps(pmc.get('accepted_by_modality', {}))}",
        ]
        if pmc.get("download_failures"):
            out.append(f"- Download failures: {', '.join(pmc['download_failures'])}")
        if pmc.get("mixed_modality_articles"):
            out.append(f"- Matched both MG and PROTAC queries (classified into one): "
                       f"{', '.join(pmc['mixed_modality_articles'])}")
        accepted_by_policy = pmc.get("policy_accepted", [])
        if accepted_by_policy:
            out += ["", "### Accepted by the DC50/Dmax mention policy", "",
                    "To veto an article, add its DOI to `reviewer_rejected_dois.txt` in this folder and push.", "",
                    "| PMCID | DOI | Type | DC50 | Dmax | Title |", "|---|---|---|---|---|---|"]
            for d in accepted_by_policy:
                out.append(f"| {d['pmcid']} | {d['doi']} | {d['article_type']} | {d['has_dc50']} | "
                           f"{d['has_dmax']} | {str(d['title']).replace('|', '/')} |")
    else:
        out.append("PMC stage did not produce a summary.")

    out += ["", "## Records", "", "| Dataset | Extracted | Post-processed | Standardized | Appended | Conflicts | Release rows |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for m in MODALITIES:
        rep = merge.get(m, {})
        out.append(
            f"| {m} | {rows[m]['extracted'] or 0} | {rows[m]['post_processed'] or 0} | {rows[m]['standardized'] or 0} "
            f"| {rep.get('appended_rows', '-')} | {sum(rep.get('conflict_rows', {}).values()) if rep else '-'} "
            f"| {rep.get('previous_rows', '-')} → {rep.get('release_rows', '-')} |"
        )
        if rep.get("new_id_range"):
            out[-1] += f" new IDs {rep['new_id_range'][0]} .. {rep['new_id_range'][1]}"

    counts = changes.get("counts", {})
    out += ["", "## Standardization mappings", "",
            f"- Added by the mapping agent: {counts.get('applied', 0)}; "
            f"not mapped: {counts.get('unresolved', 0) + counts.get('existing_entry', 0)}",
            "- Details: `standardization_mapping_require_review.md`"]

    out += ["", "## Quality checks", ""]
    if qc:
        out += [f"{qc['failed']} failed, {qc['warnings']} warnings.", "", "| Check | Status | Detail |", "|---|---|---|"]
        for check in qc["checks"]:
            detail = check["detail"] if isinstance(check["detail"], str) else json.dumps(check["detail"], ensure_ascii=False)
            out.append(f"| {check['name']} | {check['status']} | {detail.replace('|', '/')[:300]} |")
        if qc["failed"]:
            out += ["", "**The release files were restored to their pre-merge state because a check failed.**"]
    else:
        out.append("Not run.")

    out += ["", "## Reviewer checklist", "",
            "- [ ] Mappings added to `scripts/data_standardization.py` (see the review file)",
            "- [ ] Articles accepted by the DC50/Dmax policy (veto via `reviewer_rejected_dois.txt`)",
            "- [ ] `merge_conflicts.csv`",
            "- [ ] New rows in `data/release/AutoTPDplus-*.csv` and the row-count change above",
            "", "Editing the mappings or the veto list re-runs standardization and the merge in CI."]
    path = ctx.updates / "workflow_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path






def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def command_run(args: argparse.Namespace) -> int:
    today = datetime.now()
    if args.first_monday_only and not (today.weekday() == 0 and today.day <= 7):
        print("Not the first Monday of the month; nothing to do.")
        return 0
    ctx = Context(args.month, load_config(args.config), work_root=args.work_root)
    ctx.status = read_json(ctx.work / "status.json", {}) or {}
    ctx.updates.mkdir(parents=True, exist_ok=True)
    stages = STAGES[STAGES.index(args.from_stage or STAGES[0]): STAGES.index(args.to_stage) + 1]
    if args.from_stage is None:


        stages = [s for s in stages if not (s in {"pmc", "extract"} and ctx.status.get(s, {}).get("status") == "done")]

    outcome, failure, stage = "completed", None, "setup"
    try:
        if not build.load_state(ctx.state).get("next_id"):
            raise StageError(f"{ctx.state} has no next_id; run build_autotpdplus.py bootstrap first")
        for stage in stages:
            print(f"\n=== {stage} ===")
            if stage == "pmc":
                stage_pmc(ctx)
            elif stage == "extract":
                stage_extract(ctx)
            elif stage == "post":
                stage_post(ctx)
            elif stage == "standardize":
                stage_standardize(ctx, use_agent=not args.skip_agent)
            elif stage == "merge":
                snapshot_release(ctx, ctx.snapshot)
                stage_merge(ctx, ctx.snapshot)
            elif stage == "qc":
                qc = stage_qc(ctx, ctx.snapshot, ctx.work / "data_standardization_before.py",
                              reviewer_edits_allowed=False)
                if qc["failed"] and (ctx.snapshot / "complete").exists():
                    restore_release(ctx, ctx.snapshot)
                    outcome = "quality checks failed; release not updated"
                ctx.mark("qc", "done", failed=qc["failed"], warnings=qc["warnings"])
    except Exception as exc:
        ctx.mark(stage, "failed", error=str(exc))
        traceback.print_exc()
        outcome, failure = f"failed at {stage}", str(exc)
        if (ctx.snapshot / "complete").exists() and stage in {"merge", "qc"}:
            restore_release(ctx, ctx.snapshot)

    if outcome == "completed" and stages[-1] == "qc":
        state = build.load_state(ctx.state)
        state["last_successful_run"] = {"month": ctx.month, "completed_at": utc_now()}
        build.save_state(ctx.state, state)
    summary = write_summary(ctx, outcome, failure)
    print(f"\nSummary: {summary}")

    if not args.no_pr:
        cmd = [sys.executable, REPO / "scripts" / "monthly" / "prepare_monthly_pr.py",
               "--month", ctx.month, "--config", args.config]
        try:
            run_logged(cmd, ctx.log("pr"))
        except StageError as exc:
            print(f"PR preparation failed: {exc}")
            return 1
    return 0 if failure is None else 1


def command_restandardize(args: argparse.Namespace) -> int:
    ctx = Context(args.month, load_config(args.config), work_root=args.work_root)
    base = ctx.work / "base"
    snapshot_from_git(ctx, args.base_ref, base)
    script_at_base = ctx.work / "data_standardization_base.py"
    shown = subprocess.run(["git", "show", f"{args.base_ref}:{STD_SCRIPT}"], cwd=ctx.repo, capture_output=True)
    if shown.returncode == 0:
        script_at_base.write_bytes(shown.stdout)

    for stage in ("pmc", "extract", "post"):
        ctx.mark(stage, "committed earlier")
    stage_standardize(ctx, use_agent=False)
    stage_merge(ctx, base)
    qc = stage_qc(ctx, base, script_at_base if script_at_base.exists() else None, reviewer_edits_allowed=True)
    ctx.mark("qc", "done", failed=qc["failed"], warnings=qc["warnings"])
    outcome = "re-standardized in CI after review"
    if qc["failed"]:
        restore_release(ctx, base)
        outcome = "re-standardized in CI; quality checks failed, release not updated"
    write_summary(ctx, outcome)
    return 1 if qc["failed"] else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly AutoTPD+ update workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the monthly workflow.")
    r.add_argument("--month", default=datetime.now().strftime("%Y-%m"))
    r.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    r.add_argument("--from_stage", choices=STAGES, default=None,
                   help="Re-run from this stage (default: resume, skipping finished pmc/extract).")
    r.add_argument("--to_stage", choices=STAGES, default=STAGES[-1])
    r.add_argument("--skip_agent", action="store_true", help="Standardize once without the mapping agent.")
    r.add_argument("--no_pr", action="store_true", help="Dry run: do not commit, push or open a PR.")
    r.add_argument("--first_monday_only", action="store_true", help="Exit unless today is the first Monday.")
    r.add_argument("--work_root", type=Path, default=None, help="Override paths.work_root.")

    s = sub.add_parser("restandardize", help="CI: re-run standardization and merge after a mapping review.")
    s.add_argument("--month", required=True)
    s.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    s.add_argument("--base_ref", required=True, help="e.g. origin/under_review")
    s.add_argument("--work_root", type=Path, default=None, help="Override paths.work_root (CI runners).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return command_run(args) if args.command == "run" else command_restandardize(args)


if __name__ == "__main__":
    sys.exit(main())
