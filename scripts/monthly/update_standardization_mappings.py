#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

KIND_MAPPING = {
    "cell_line": "CL_SEARCH_MAP",
    "cell_line_modification": "CL_MODIFICATION_MAP",
    "degradation_target": "POI_LOOKUP",
    "recruiter": "E3_LOOKUP",
    "assay": "ASSAY_STD_MAP",
}
APPLY_CONFIDENCE = {"high", "medium"}
LLM_BATCH_SIZE = 25

KIND_INSTRUCTIONS = {
    "cell_line": (
        "Map an extracted cell-line string to the Cellosaurus entry it denotes. Give the "
        "Cellosaurus accession (CVCL_xxxx) when you know it, otherwise the exact Cellosaurus "
        "recommended name. Engineered derivatives map to their parental line (the construct is "
        "recorded separately). Use decision 'no_standard' for primary cells, tissues, organoids, "
        "in vivo models or cell-free systems that have no Cellosaurus entry."
    ),
    "cell_line_modification": (
        "Decide whether the extracted cell-line string states a genetic engineering of the line: "
        "reporter tags (HiBiT, nLuc, GFP...), transgenes, knock-ins/knock-outs, silencing, chimeric "
        "receptors, or a drug-resistant derivative. If so, 'mapping' is the construct text exactly as "
        "it appears in the string (e.g. 'HEK-293T (BTK-nLuc)' -> 'BTK-nLuc'). Subclones, xenografts, "
        "co-cultures and patient genotypes are not engineering: use decision 'no_modification'."
    ),
    "degradation_target": (
        "Map an extracted degradation-target string to {\"search\": <HGNC gene symbol>, "
        "\"mutation\": <point mutation like G466V or null>}. Use decision 'no_standard' with "
        "mapping {\"search\": null, \"mutation\": null} when the target is not a single protein "
        "(e.g. a fluorescent reporter alone)."
    ),
    "recruiter": (
        "Map an extracted E3 ligase / recruiter string to the HGNC gene symbol used as the search "
        "term (e.g. 'CRL4-CRBN' -> 'CRBN', 'cIAP1' -> 'BIRC2'). Use decision 'no_standard' when "
        "no recruiter is identifiable."
    ),
    "assay": (
        "Map an extracted assay description to one of the existing technique families listed "
        "under allowed_values. Never invent a new family; if none fits, use decision 'unresolved' "
        "and name the family you would suggest in the rationale. Use 'no_standard' when the text "
        "is not a protein-level degradation assay."
    ),
}


def load_module(path: Path, name: str = "data_standardization_under_edit") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def merge_reports(paths: list[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        dataset = path.parent.name
        for item in report.get("items", []):
            key = (item["kind"], item["value"])
            entry = merged.setdefault(key, {**item, "rows": 0, "examples": [], "datasets": []})
            entry["rows"] += int(item.get("rows", 0))
            entry["examples"] = (entry["examples"] + item.get("examples", []))[:3]
            if dataset not in entry["datasets"]:
                entry["datasets"].append(dataset)
    return [merged[key] for key in sorted(merged)]


def build_prompt(kind: str, items: list[dict[str, Any]], module: ModuleType) -> str:
    mapping = getattr(module, KIND_MAPPING[kind])
    style = dict(list(mapping.items())[:12])
    payload = {
        "task": KIND_INSTRUCTIONS[kind],
        "mapping_name": KIND_MAPPING[kind],
        "existing_entries_for_style": style,
        "values": [
            {
                "value": item["value"],
                "rows": item["rows"],
                "examples": item["examples"],
                "close_existing": item.get("close_existing", []),
            }
            for item in items
        ],
    }
    if kind == "assay":
        payload["allowed_values"] = sorted({v for v in mapping.values() if v})
    return (
        "You curate the standardization mappings of a targeted protein degradation dataset.\n"
        "For each value below return one JSON object with keys: value (copied exactly), "
        "decision ('map', 'no_standard', 'no_modification' or 'unresolved'), mapping, "
        "confidence ('high', 'medium' or 'low') and rationale (one sentence citing the evidence).\n"
        "Be conservative: when unsure use 'unresolved' or confidence 'low'.\n"
        "Return only a JSON array, no commentary.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def parse_llm_json(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("LLM response contains no JSON array")
    data = json.loads(cleaned[start:end + 1])
    return [item for item in data if isinstance(item, dict)]


def openrouter_proposer(model_name: str | None = None) -> Callable[[str], str]:
    from dotenv import load_dotenv
    from langchain_community.chat_models import ChatOpenAI

    load_dotenv()
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        model_name=model_name or os.getenv("MAPPING_LLM_MODEL_NAME") or os.getenv("LLM_MODEL_NAME"),
        temperature=0,
        request_timeout=600,
    )
    return lambda prompt: llm.invoke(prompt).content


def propose(
    items: list[dict[str, Any]],
    module: ModuleType,
    ask: Callable[[str], str],
) -> dict[tuple[str, str], dict[str, Any]]:
    proposals: dict[tuple[str, str], dict[str, Any]] = {}
    for kind in KIND_MAPPING:
        kind_items = [item for item in items if item["kind"] == kind]
        for start in range(0, len(kind_items), LLM_BATCH_SIZE):
            chunk = kind_items[start:start + LLM_BATCH_SIZE]
            wanted = {item["value"] for item in chunk}
            try:
                answers = parse_llm_json(ask(build_prompt(kind, chunk, module)))
            except Exception as exc:
                print(f"  LLM proposal failed for {kind}: {exc}")
                continue
            for answer in answers:
                value = answer.get("value")
                if value in wanted:
                    proposals[(kind, value)] = answer
    return proposals


def verify(kind: str, value: str, proposal: dict[str, Any], module: ModuleType) -> tuple[bool, Any, str]:
    decision = proposal.get("decision")
    mapping = proposal.get("mapping")

    if kind == "cell_line":
        if decision == "no_standard":
            return True, None, "explicit no-standard entry"
        if decision != "map" or not isinstance(mapping, str) or not mapping.strip():
            return False, None, "no usable proposal"
        term = mapping.strip()
        result = module.cellosaurus_fetch(term) if term.startswith("CVCL_") else module.cellosaurus_search(term)
        if not result:
            return False, None, f"Cellosaurus does not resolve '{term}'"
        return True, term, f"Cellosaurus {result.get('accession')} {result.get('name')} ({result.get('species')})"

    if kind == "cell_line_modification":
        if decision != "map" or not isinstance(mapping, str) or not mapping.strip():
            return False, None, "no modification proposed"
        if mapping.strip().lower() not in value.lower():
            return False, None, "construct text does not appear in the cell-line string"
        return True, mapping.strip(), "construct text found in the extracted string"

    if kind == "degradation_target":
        if decision == "no_standard":
            return True, {"search": None, "mutation": None}, "explicit no-standard entry"
        if decision != "map" or not isinstance(mapping, dict) or not mapping.get("search"):
            return False, None, "no usable proposal"
        search, mutation = str(mapping["search"]).strip(), mapping.get("mutation") or None
        if mutation is not None and not module.parse_mutation(str(mutation)):
            return False, None, f"mutation '{mutation}' is not a point mutation like G466V"
        gene, source = module.find_gene_symbol(search, module.HUMAN)
        if not gene:
            return False, None, f"HGNC/UniProt do not resolve '{search}'"
        return True, {"search": search, "mutation": mutation}, f"{source} -> {gene}"

    if kind == "recruiter":
        if decision == "no_standard":
            return True, None, "explicit no-standard entry"
        if decision != "map" or not isinstance(mapping, str) or not mapping.strip():
            return False, None, "no usable proposal"
        gene, source = module.find_gene_symbol(mapping.strip(), module.HUMAN)
        if not gene:
            return False, None, f"HGNC/UniProt do not resolve '{mapping}'"
        return True, mapping.strip(), f"{source} -> {gene}"

    if kind == "assay":
        if decision == "no_standard":
            return True, None, "explicit no-standard entry"
        families = {v for v in module.ASSAY_STD_MAP.values() if v}
        if decision != "map" or mapping not in families:
            return False, None, f"'{mapping}' is not an existing assay family"
        return True, mapping, "existing assay family"

    return False, None, f"unknown kind {kind}"


def decide(
    items: list[dict[str, Any]],
    proposals: dict[tuple[str, str], dict[str, Any]],
    module: ModuleType,
) -> list[dict[str, Any]]:
    changes = []
    for item in items:
        kind, value = item["kind"], item["value"]
        mapping_name = KIND_MAPPING[kind]
        base = {
            "kind": kind, "value": value, "mapping_name": mapping_name, "rows": item["rows"],
            "datasets": item.get("datasets", []), "examples": item["examples"],
        }
        if value in getattr(module, mapping_name):
            changes.append({**base, "status": "existing_entry",
                            "reason": "key already present but did not resolve; left unchanged"})
            continue
        proposal = proposals.get((kind, value))
        if proposal is None:
            changes.append({**base, "status": "unresolved", "reason": "no LLM proposal"})
            continue
        confidence = str(proposal.get("confidence", "low")).lower()
        rationale = str(proposal.get("rationale", ""))
        if kind == "cell_line_modification" and proposal.get("decision") == "no_modification":
            changes.append({**base, "status": "no_modification", "confidence": confidence,
                            "rationale": rationale})
            continue
        ok, written, note = verify(kind, value, proposal, module)
        record = {**base, "proposed": proposal.get("mapping"), "decision": proposal.get("decision"),
                  "confidence": confidence, "rationale": rationale, "verification": note}
        if ok and confidence in APPLY_CONFIDENCE:
            changes.append({**record, "status": "applied", "mapping": written})
        else:
            reason = note if not ok else f"confidence {confidence}"
            changes.append({**record, "status": "unresolved", "reason": reason})
    return changes


def format_literal(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{format_literal(k)}: {format_literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"cannot write {type(value).__name__} into a mapping")


def find_dict_spans(source: str) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for node in ast.parse(source).body:
        target, value = None, None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target and isinstance(value, ast.Dict):
            found[target] = (node.lineno - 1, value.end_lineno - 1)
    return found


def patch_source(source: str, changes: list[dict[str, Any]], run_label: str) -> tuple[str, dict[str, list[int]]]:
    applied = [c for c in changes if c["status"] == "applied"]
    lines = source.splitlines(keepends=True)
    closing = {name: span[1] for name, span in find_dict_spans(source).items()}
    by_mapping: dict[str, list[dict[str, Any]]] = {}
    for change in applied:
        by_mapping.setdefault(change["mapping_name"], []).append(change)

    for mapping_name in sorted(by_mapping, key=lambda name: closing[name], reverse=True):
        index = closing[mapping_name]
        if lines[index].strip() != "}":
            raise ValueError(f"{mapping_name} does not end with '}}' on its own line")
        block = [f"    # --- added by monthly mapping agent ({run_label}); see standardization_mapping_require_review.md ---\n"]
        for change in sorted(by_mapping[mapping_name], key=lambda c: c["value"]):
            block.append(f"    {format_literal(change['value'])}: {format_literal(change['mapping'])},\n")
        if index > 0 and not lines[index - 1].rstrip().endswith((",", "{")) and lines[index - 1].strip() and not lines[index - 1].lstrip().startswith("#"):
            lines[index - 1] = lines[index - 1].rstrip("\n") + ",\n"
        lines[index:index] = block

    patched = "".join(lines)
    key_lines: dict[str, list[int]] = {}
    patched_lines = patched.splitlines()
    spans = find_dict_spans(patched)
    for change in applied:
        prefix = f"    {format_literal(change['value'])}:"
        first, last = spans[change["mapping_name"]]
        key_lines[f"{change['mapping_name']}::{change['value']}"] = [
            number + 1 for number in range(first, last) if patched_lines[number].startswith(prefix)
        ]
    return patched, key_lines


def check_only_additions(before: ModuleType, after: ModuleType, changes: list[dict[str, Any]]) -> list[str]:
    errors = []
    expected: dict[str, dict[str, Any]] = {}
    for change in changes:
        if change["status"] == "applied":
            expected.setdefault(change["mapping_name"], {})[change["value"]] = change["mapping"]
    for mapping_name in KIND_MAPPING.values():
        old, new = getattr(before, mapping_name), getattr(after, mapping_name)
        for key, value in old.items():
            if key not in new:
                errors.append(f"{mapping_name}: existing key {key!r} removed")
            elif new[key] != value:
                errors.append(f"{mapping_name}: existing key {key!r} changed")
        added = {key: new[key] for key in new if key not in old}
        if added != expected.get(mapping_name, {}):
            errors.append(f"{mapping_name}: added entries differ from the change list")
    return errors


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return str(text).replace("|", "\\|").replace("\n", " ")


def _examples(change: dict[str, Any]) -> str:
    parts = []
    for example in change.get("examples", [])[:2]:
        doi = example.get("DOI", "")
        compound = example.get("Compound_Name", "")
        parts.append(f"{doi} {compound}".strip())
    return "; ".join(parts)


def render_review(changes: list[dict[str, Any]], key_lines: dict[str, list[int]], run_label: str, script: str) -> str:
    applied = [c for c in changes if c["status"] == "applied"]
    unresolved = [c for c in changes if c["status"] == "unresolved"]
    existing = [c for c in changes if c["status"] == "existing_entry"]
    no_modification = [c for c in changes if c["status"] == "no_modification"]

    out = [
        f"# Standardization mapping review: {run_label}",
        "",
        f"The monthly mapping agent added the entries below to `{script}` on this branch.",
        "Each entry was proposed by the LLM, verified as noted, and applied only with high or medium confidence.",
        "To correct one, edit or delete that line in the script and push; CI re-runs standardization and the merge.",
        "",
        "| Mapping | Added | Not mapped |",
        "|---|---:|---:|",
    ]
    for mapping_name in KIND_MAPPING.values():
        n_added = sum(c["mapping_name"] == mapping_name for c in applied)
        n_open = sum(c["mapping_name"] == mapping_name for c in unresolved + existing)
        out.append(f"| `{mapping_name}` | {n_added} | {n_open} |")

    out += ["", "## Added mappings", ""]
    if not applied:
        out.append("None.")
    for mapping_name in KIND_MAPPING.values():
        group = [c for c in applied if c["mapping_name"] == mapping_name]
        if not group:
            continue
        out += [f"### `{mapping_name}`", "",
                "| Line | Value | Mapping | Rows | Confidence | Verification | Rationale |",
                "|---:|---|---|---:|---|---|---|"]
        for change in sorted(group, key=lambda c: c["value"]):
            lines = key_lines.get(f"{mapping_name}::{change['value']}", [])
            out.append(
                f"| {','.join(map(str, lines))} | `{_cell(change['value'])}` | `{_cell(change['mapping'])}` "
                f"| {change['rows']} | {change['confidence']} | {_cell(change['verification'])} "
                f"| {_cell(change['rationale'])} |"
            )
        out.append("")

    out += ["", "## Not mapped (needs a human decision)", ""]
    if not unresolved and not existing:
        out.append("None.")
    else:
        out += ["| Mapping | Value | Rows | Reason | LLM suggestion | Examples |", "|---|---|---:|---|---|---|"]
        for change in sorted(unresolved + existing, key=lambda c: (c["mapping_name"], c["value"])):
            suggestion = change.get("proposed")
            out.append(
                f"| `{change['mapping_name']}` | `{_cell(change['value'])}` | {change['rows']} "
                f"| {_cell(change.get('reason', ''))} | {_cell(suggestion) if suggestion is not None else ''} "
                f"| {_cell(_examples(change))} |"
            )

    if no_modification:
        out += ["", "## Cell-line strings judged to have no engineering", "",
                "| Value | Rows | Rationale |", "|---|---:|---|"]
        for change in sorted(no_modification, key=lambda c: c["value"]):
            out.append(f"| `{_cell(change['value'])}` | {change['rows']} | {_cell(change.get('rationale', ''))} |")
    return "\n".join(out) + "\n"


def run(
    report_paths: list[Path],
    script_path: Path,
    run_label: str,
    changes_json: Path,
    review_md: Path,
    ask: Callable[[str], str] | None,
) -> dict[str, Any]:
    items = merge_reports(report_paths)
    before = load_module(script_path, "data_standardization_before")
    proposals = propose([i for i in items if i["value"] not in getattr(before, KIND_MAPPING[i["kind"]])],
                        before, ask) if (items and ask) else {}
    changes = decide(items, proposals, before)

    original = script_path.read_text(encoding="utf-8")
    patched, key_lines = patch_source(original, changes, run_label)
    if patched != original:
        script_path.write_text(patched, encoding="utf-8")
        try:
            errors = check_only_additions(before, load_module(script_path, "data_standardization_after"), changes)
        except Exception as exc:
            errors = [f"patched script does not import: {exc}"]
        if errors:
            script_path.write_text(original, encoding="utf-8")
            raise RuntimeError("mapping patch rejected and reverted: " + "; ".join(errors))

    summary = {
        "run_label": run_label,
        "script": str(script_path),
        "counts": {status: sum(c["status"] == status for c in changes)
                   for status in ("applied", "unresolved", "existing_entry", "no_modification")},
        "changes": changes,
        "key_lines": key_lines,
    }
    changes_json.parent.mkdir(parents=True, exist_ok=True)
    changes_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    review_md.parent.mkdir(parents=True, exist_ok=True)
    review_md.write_text(render_review(changes, key_lines, run_label, script_path.name), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose, verify and add standardization mappings.")
    parser.add_argument("--unmapped_report", type=Path, nargs="+", required=True)
    parser.add_argument("--standardization_script", type=Path, default=Path("scripts/data_standardization.py"))
    parser.add_argument("--run_label", required=True)
    parser.add_argument("--changes_json", type=Path, required=True)
    parser.add_argument("--review_md", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Override MAPPING_LLM_MODEL_NAME / LLM_MODEL_NAME.")
    parser.add_argument("--no_llm", action="store_true", help="Only report unmapped values; add nothing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ask = None if args.no_llm else openrouter_proposer(args.model)
    try:
        summary = run(args.unmapped_report, args.standardization_script, args.run_label,
                      args.changes_json, args.review_md, ask)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    print(json.dumps(summary["counts"]))
    return 0

if __name__ == "__main__":
    sys.exit(main())
