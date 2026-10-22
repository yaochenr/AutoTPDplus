

















import json
import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

try:



    from gepa.core.adapter import EvaluationBatch
except Exception:
    EvaluationBatch = None



EDITABLE_SECTIONS = ("Goal", "Extraction Principles")


PAUSE_MARKER = "PAUSE.json"
PAUSE_REVIEW = "REVIEW.md"
CONTINUE_SIGNAL = "CONTINUE"


def cache_index(cache_file) -> Dict:





    p = Path(cache_file)
    if not p.exists():
        return {}
    try:
        cache = json.loads(p.read_text())
    except Exception:
        return {}
    idx = {}
    for doi, entries in cache.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            key = (doi, e.get("feature"),
                   str(e.get("pred_value")), str(e.get("labeled_value")))
            idx[key] = str(e.get("result"))
    return idx






def parse_prompt_sections(prompt_text: str) -> List[Dict[str, str]]:





    lines = prompt_text.splitlines(keepends=True)
    sections: List[Dict[str, str]] = []
    current = {"name": "", "header_line": "", "body": ""}

    for line in lines:
        m = re.match(r"^#\s+(.+?)\s*$", line.rstrip("\n"))
        if m:
            if current["header_line"] or current["body"]:
                sections.append(current)
            current = {"name": m.group(1).strip(), "header_line": line, "body": ""}
        else:
            current["body"] += line

    if current["header_line"] or current["body"]:
        sections.append(current)
    return sections


def prompt_to_candidate(prompt_text: str) -> Dict[str, str]:

    candidate: Dict[str, str] = {}
    for sec in parse_prompt_sections(prompt_text):
        if sec["name"] in EDITABLE_SECTIONS:


            candidate[sec["name"]] = sec["body"].strip("\n")
    if not candidate:
        raise ValueError(
            f"Seed prompt has none of the editable sections {EDITABLE_SECTIONS}. "
            "Check the prompt's '# Goal' / '# Extraction Principles' headers."
        )
    return candidate


def candidate_to_prompt(candidate: Dict[str, str], template_prompt: str) -> str:





    out: List[str] = []
    for sec in parse_prompt_sections(template_prompt):
        out.append(sec["header_line"])
        if sec["name"] in candidate:
            body = candidate[sec["name"]].strip("\n")


            out.append((body + "\n\n") if body else "\n")
        else:
            out.append(sec["body"])
    return "".join(out)






_GT_KEY_FIELDS = (
    "Compound_Name", "Degradation_Target", "Recruiter", "Assay", "Cell_Line",
    "DC50", "DC50_units", "DC50_h", "Dmax", "Dmax_h", "Dmax_conc",
)


def _clean(value: Any) -> Any:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else value


def build_paper_feedback(
    capo,
    doi: str,
    paper_dir: Any,
    metrics: Dict[str, float],
) -> str:





    paper_dir = Path(paper_dir)
    all_ground_truth = capo._load_ground_truth_for_doi(doi)

    pred_csv = paper_dir / "data_processed.csv"
    all_predictions = (
        pd.read_csv(pred_csv).to_dict("records") if pred_csv.exists() else []
    )

    detailed_eval_file = paper_dir / "evaluation_detailed_by_doi.json"
    detailed_evaluation: Dict[str, Any] = {}
    if detailed_eval_file.exists():
        try:
            detailed_evaluation = json.loads(detailed_eval_file.read_text())
        except Exception:
            detailed_evaluation = {}

    classified = capo._classify_errors_by_type(
        doi=doi,
        detailed_evaluation=detailed_evaluation,
        all_ground_truth=all_ground_truth,
        all_predictions=all_predictions,
    )
    g1 = classified["group1_record_level_omissions"]
    g2 = classified["group2_unmatched_extra_data"]

    parts: List[str] = []
    parts.append(
        "Extraction scored against ground truth for this paper (0-1, higher is better):\n"
        f"  record-level precision={metrics['part1_precision']:.2f}, "
        f"recall={metrics['part1_recall']:.2f}\n"
        f"  field-level  precision={metrics['part2_precision']:.2f}, "
        f"recall={metrics['part2_recall']:.2f}"
    )

    parts.append(
        f"\nMISSED records (False Negatives - present in ground truth but NOT "
        f"extracted): {g1['count']}"
    )
    for i, e in enumerate(g1["errors"], 1):
        gt = e.get("gt_full_record", {})
        parts.append(
            f"  FN #{i}: "
            + json.dumps({k: _clean(gt.get(k, "")) for k in _GT_KEY_FIELDS}, ensure_ascii=False)
        )

    parts.append(
        f"\nWRONG / EXTRA records (False Positives - extracted but unmatched, or "
        f"matched on identity yet with wrong measurement fields): {g2['count']}"
    )
    for i, e in enumerate(g2["errors"], 1):
        ex = e.get("extracted", {})
        line = (
            f"  FP #{i} ({e.get('error_type', 'unknown')}): "
            + json.dumps({k: _clean(v) for k, v in ex.items()}, ensure_ascii=False)
        )
        if e.get("reason"):
            line += f"\n      reason: {e['reason']}"
        if e.get("ground_truth"):
            line += "\n      correct value: " + json.dumps(
                {k: _clean(v) for k, v in e["ground_truth"].items()}, ensure_ascii=False
            )
        parts.append(line)

    if g1["count"] == 0 and g2["count"] == 0:
        parts.append("\nNo errors on this paper - extraction was perfect.")
    else:
        parts.append(
            "\nWrite prompt guidance that would prevent the above errors in general. "
            "Do NOT hardcode these specific compounds/targets/cell lines; fix the root cause."
        )
    return "\n".join(parts)






class _EvalBatchFallback:


    def __init__(self, outputs, scores, trajectories):
        self.outputs = outputs
        self.scores = scores
        self.trajectories = trajectories
        self.objective_scores = None
        self.num_metric_calls = None


class TpdExtractionAdapter:





    propose_new_texts = None

    def __init__(self, capo, template_prompt: str, work_dir: Any,
                 pause_on_new_matches: bool = False, pause_dir: Any = None,
                 pause_context: Optional[Dict[str, Any]] = None,
                 poll_interval: float = 3.0):
        self.capo = capo
        self.template_prompt = template_prompt
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)


        self._cache: Dict[str, Dict[str, Any]] = {}






        self.pause_on_new_matches = pause_on_new_matches
        self.pause_dir = Path(pause_dir) if pause_dir else (self.work_dir / ".gepa_pause")
        self.pause_context = pause_context or {}
        self.poll_interval = poll_interval
        self.cache_file = self.capo.config.get("evaluation", {}).get("cache_file_path")

    @staticmethod
    def _hash_candidate(candidate: Dict[str, str]) -> str:
        blob = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _scalarize(metrics: Dict[str, float]) -> float:
        return (
            metrics["part1_precision"]
            + metrics["part1_recall"]
            + metrics["part2_precision"]
            + metrics["part2_recall"]
        ) / 4.0

    def _rollout(self, candidate: Dict[str, str], doi: str, pmc_id) -> Dict[str, Any]:
        chash = self._hash_candidate(candidate)
        key = f"{chash}::{doi}"
        if key in self._cache:
            return self._cache[key]

        out_dir = self.work_dir / chash
        out_dir.mkdir(parents=True, exist_ok=True)







        safe_doi = doi.replace("/", "_").replace(":", "_")
        self.capo.config["data"]["output_dir"] = str(out_dir / f"clean_{safe_doi}")


        try:
            from tpd_curator import pipeline as _pipeline
            _pipeline.store.clear()
        except Exception:
            pass

        result = self.capo._extract_paper(
            doi=doi,
            pmc_id=pmc_id,
            output_dir=out_dir,
            round_dir=out_dir,
        )
        self._cache[key] = result
        return result

    def evaluate(
        self,
        batch: List[Dict[str, Any]],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ):
        full_prompt = candidate_to_prompt(candidate, self.template_prompt)
        self.capo.current_prompt = full_prompt

        zero = {k: 0.0 for k in (
            "part1_precision", "part1_recall", "part2_precision", "part2_recall")}


        cache_before = cache_index(self.cache_file) if self.pause_on_new_matches else {}

        results: List[Dict[str, Any]] = []
        for inst in batch:
            doi = inst["doi"]
            pmc_id = inst.get("pmc_id")
            try:
                result = self._rollout(candidate, doi, pmc_id)
                results.append({"doi": doi, "metrics": result["metrics"],
                                "paper_dir": result.get("paper_dir", ""),
                                "error": None, "result": result})
            except Exception as e:
                results.append({"doi": doi, "metrics": dict(zero),
                                "paper_dir": "", "error": str(e), "result": None})




        if self.pause_on_new_matches:
            new_entries = self._new_cache_entries(cache_before)
            if new_entries:
                self._pause_for_review(new_entries, results)
                self._reevaluate_after_review(results)

        outputs: List[Dict[str, Any]] = []
        scores: List[float] = []
        trajectories: Optional[List[Dict[str, Any]]] = [] if capture_traces else None
        for r in results:
            scores.append(self._scalarize(r["metrics"]))
            outputs.append({"doi": r["doi"], "metrics": r["metrics"],
                            "paper_dir": r["paper_dir"], "error": r["error"]})
            if capture_traces:
                trajectories.append({"doi": r["doi"], "metrics": r["metrics"],
                                     "paper_dir": r["paper_dir"], "error": r["error"]})

        if EvaluationBatch is not None:
            return EvaluationBatch(outputs=outputs, scores=scores,
                                   trajectories=trajectories)
        return _EvalBatchFallback(outputs, scores, trajectories)




    def _new_cache_entries(self, cache_before: Dict) -> List[Dict[str, Any]]:

        after = cache_index(self.cache_file)
        out = []
        for (doi, feat, pred, lab), res in after.items():
            old = cache_before.get((doi, feat, pred, lab))
            if old is None:
                out.append({"doi": doi, "feature": feat, "pred_value": pred,
                            "labeled_value": lab, "result": res, "change": "new"})
            elif old != res:
                out.append({"doi": doi, "feature": feat, "pred_value": pred,
                            "labeled_value": lab, "result": res,
                            "change": f"{old}->{res}"})
        return out

    def _write_review_markdown(self, new_entries, results, path: Path):
        lines = ["# GEPA paused — review these NEW semantic-match judgments\n",
                 "\nThe evaluation below produced new (uncurated) matches. A wrong "
                 "'no' fails the whole record. Fix wrong `result` values in the cache:",
                 f"\n    {self.cache_file}\n",
                 "\nThen, in another terminal:",
                 "\n    pixi run python scripts/prompt_optimization/gepa_continue.py --verify",
                 "\n    pixi run python scripts/prompt_optimization/gepa_continue.py\n",
                 "\n## Current (uncurated) per-paper scores\n"]
        for r in results:
            m = r["metrics"]
            lines.append(f"- {r['doi']}: p1={m['part1_precision']:.2f}/{m['part1_recall']:.2f} "
                         f"p2={m['part2_precision']:.2f}/{m['part2_recall']:.2f}"
                         + (f"  ERROR: {r['error']}" if r["error"] else ""))
        lines.append("\n## New judgments this evaluation\n")
        for e in new_entries:
            tag = "" if e["change"] == "new" else f" (CHANGED {e['change']})"
            lines.append(f"\nDOI {e['doi']}{tag}\n  [{e['feature']}] result={e['result']!r}"
                         f"\n     pred = {e['pred_value']!r}\n     vs gt= {e['labeled_value']!r}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _pause_for_review(self, new_entries, results):
        self.pause_dir.mkdir(parents=True, exist_ok=True)

        sig = self.pause_dir / CONTINUE_SIGNAL
        if sig.exists():
            sig.unlink()

        papers = [{"doi": r["doi"], "paper_dir": r["paper_dir"]}
                  for r in results if r["paper_dir"]]
        marker = {
            "paused_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_file": str(self.cache_file),
            "papers": papers,
            "num_new_entries": len(new_entries),
            **self.pause_context,
        }
        (self.pause_dir / PAUSE_MARKER).write_text(
            json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_review_markdown(new_entries, results, self.pause_dir / PAUSE_REVIEW)

        print("\n" + "=" * 80)
        print(f"⏸  GEPA PAUSED — {len(new_entries)} new semantic-match judgment(s) to review")
        print("=" * 80)
        print(f"Review file : {self.pause_dir / PAUSE_REVIEW}")
        print(f"Cache file  : {self.cache_file}")
        print("In another terminal:")
        print("  pixi run python scripts/prompt_optimization/gepa_continue.py --verify")
        print("  pixi run python scripts/prompt_optimization/gepa_continue.py")
        print("=" * 80, flush=True)


        while not sig.exists():
            time.sleep(self.poll_interval)

        sig.unlink()
        for fn in (PAUSE_MARKER, PAUSE_REVIEW):
            fp = self.pause_dir / fn
            if fp.exists():
                fp.unlink()
        print("▶  Resuming GEPA with corrected cache ...", flush=True)

    def _reevaluate_after_review(self, results):


        for r in results:
            if not r["paper_dir"] or r["error"]:
                continue
            try:
                ev = self.capo._re_evaluate_paper(paper_dir=Path(r["paper_dir"]), doi=r["doi"])
                metrics = self.capo._extract_metrics_from_evaluation(ev, r["doi"])
                r["metrics"] = metrics
                if r.get("result") is not None:
                    r["result"]["metrics"] = metrics
            except Exception as e:
                print(f"  [re-eval] {r['doi']} failed: {e}")

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch,
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records: List[Dict[str, Any]] = []
        for traj in (eval_batch.trajectories or []):
            doi = traj["doi"]
            if traj.get("error"):
                feedback = (
                    f"Extraction failed with an error: {traj['error']}. "
                    "The prompt may be producing malformed or unparseable output."
                )
            else:
                feedback = build_paper_feedback(
                    self.capo, doi, traj["paper_dir"], traj["metrics"]
                )
            records.append({
                "Inputs": f"A scientific paper about molecular glues (DOI: {doi}).",
                "Generated Outputs": (
                    "Extraction was scored against ground truth; per-record "
                    f"errors are detailed in Feedback (score={self._scalarize(traj['metrics']):.2f})."
                ),
                "Feedback": feedback,
            })


        return {comp: records for comp in components_to_update}
