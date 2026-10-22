

import json
import yaml
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pandas as pd

from .pipeline import (
    run_llm_pipeline,
    save_csv,
    process_units,
    run_evaluation,
    clean_json_papers_batch,
)
from .utils.pause_manager import (
    PauseManager,
    PauseState,
    create_pause_state_for_batch,
)


class PipelinePausedException(Exception):

    def __init__(self, message, round_num, round_dir, aggregate_metrics):
        super().__init__(message)
        self.round_num = round_num
        self.round_dir = round_dir
        self.aggregate_metrics = aggregate_metrics


class Capo:


    def __init__(
        self,
        config_path: str,
        initial_prompt_path: str,
        batch_size: Optional[int] = None,
        precision_threshold: Optional[float] = None,
        recall_threshold: Optional[float] = None,
        with_history: bool = True,
        pause_after_extraction: bool = False,
        resume: bool = False,
        experiment_dir_to_resume: Optional[str] = None,
        verify_only: bool = False,
    ):

        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.initial_prompt_path = Path(initial_prompt_path)
        self.initial_prompt = self.initial_prompt_path.read_text(encoding="utf-8")
        self.current_prompt = self.initial_prompt

        self.batch_size = batch_size if batch_size is not None else 6
        self.precision_threshold = precision_threshold if precision_threshold is not None else 0.9
        self.recall_threshold = recall_threshold if recall_threshold is not None else 0.9
        self.max_rounds = 8

        self.with_history = with_history

        self.pause_after_extraction = pause_after_extraction
        self.resume = resume
        self.verify_only = verify_only

        results_base_dir = Path(self.config["results"]["results_dir"])

        if resume:
            if experiment_dir_to_resume:
                self.experiment_dir = Path(experiment_dir_to_resume)
            else:
                existing_dirs = sorted(
                    results_base_dir.glob("active_prompting_batch_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if not existing_dirs:
                    raise ValueError("No experiment directory found to resume. Please specify --experiment-dir")
                self.experiment_dir = existing_dirs[0]

            if not self.experiment_dir.exists():
                raise ValueError(f"Experiment directory not found: {self.experiment_dir}")

            print(f"Resuming from experiment directory: {self.experiment_dir}")
        else:
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            self.experiment_dir = results_base_dir / f"active_prompting_batch_{timestamp}"
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

        if resume:
            prompt_files = sorted(self.experiment_dir.glob("*_prompt.md"))
            if prompt_files:
                latest_prompt_file = prompt_files[-1]
                self.prompt_version = int(latest_prompt_file.stem.split("_")[0])
                self.current_prompt = latest_prompt_file.read_text(encoding="utf-8")
                print(f"Loaded prompt version {self.prompt_version:02d}: {latest_prompt_file}")
            else:
                self.prompt_version = 0
                self.current_prompt = self.initial_prompt
                print("Warning: No prompt files found in experiment directory, using initial prompt")
        else:
            self.prompt_version = 0
            initial_prompt_file = self.experiment_dir / "00_prompt.md"
            initial_prompt_file.write_text(self.initial_prompt, encoding="utf-8")
            print(f"Initial prompt saved: {initial_prompt_file}")

        print(f"\n{'='*80}")
        print(f"Batch Active Prompting Pipeline Initialized")
        print(f"{'='*80}")
        print(f"Config: {self.config_path}")
        print(f"Initial Prompt: {self.initial_prompt_path}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Precision Threshold: {self.precision_threshold}")
        print(f"Recall Threshold: {self.recall_threshold}")
        print(f"With History (3-rounds): {self.with_history}")
        print(f"Experiment Directory: {self.experiment_dir}")
        print(f"Pause After Extraction: {self.pause_after_extraction}")
        print(f"Resume: {self.resume}")
        print(f"{'='*80}\n")

    def _load_config(self) -> Dict[str, Any]:

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def _get_ground_truth_paper_list(self) -> List[Dict[str, str]]:

        filter_file = self.config["data"].get("paper_filter_file")
        if filter_file and Path(filter_file).exists():
            with open(filter_file, "r", encoding="utf-8") as f:
                papers = json.load(f)
            if papers and isinstance(papers[0], dict):
                return papers
            else:
                return [{"doi": doi, "pmc_id": None} for doi in papers]

        labeled_data_path = self.config["evaluation"]["labeled_data_path"]
        df = pd.read_csv(labeled_data_path)
        unique_dois = df["DOI"].unique().tolist()
        return [{"doi": doi, "pmc_id": None} for doi in unique_dois]

    def run_batch_round(self, round_num: int = 1) -> Dict[str, Any]:

        expected_prompt_version = round_num - 1
        prompt_file = self.experiment_dir / f"{expected_prompt_version:02d}_prompt.md"

        if prompt_file.exists():
            self.current_prompt = prompt_file.read_text(encoding="utf-8")
            self.prompt_version = expected_prompt_version
            print(f"[PROMPT] Loaded prompt version {self.prompt_version:02d} for Round {round_num}: {prompt_file.name}")
        else:
            print(f"[PROMPT] Warning: Expected prompt file not found: {prompt_file}")
            print(f"[PROMPT] Using currently loaded prompt (version {self.prompt_version:02d})")

        round_dir = self.experiment_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)
        batch_results_dir = round_dir / "batch_results"
        batch_results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%y%m%d_%H%M")

        print(f"\n{'='*80}")
        print(f"Starting Round {round_num}")
        print(f"{'='*80}")
        print(f"Round Directory: {round_dir}")
        print(f"Timestamp: {timestamp}\n")

        config_save_path = round_dir / "config.yaml"
        shutil.copy(self.config_path, config_save_path)

        pause_manager = PauseManager(results_dir=str(round_dir))

        all_papers = self._get_ground_truth_paper_list()
        batch_papers = all_papers[:self.batch_size]

        print(f"Total available papers: {len(all_papers)}")
        print(f"Processing batch of {len(batch_papers)} papers\n")

        pause_state = pause_manager.load_pause_state()

        if pause_state and pause_state.batch_info:
            pause_round = pause_state.round

            if pause_round == round_num:
                print(f"\n[RESUME] Within-round resume (Round {round_num})")
                print(f"[RESUME] Found pause state from {pause_state.timestamp}")
                completed_papers = pause_state.batch_info["completed_papers"]
                last_paper_index = pause_state.batch_info["paper_index_in_batch"]

                print(f"[RESUME] Papers completed: {len(completed_papers)}/{len(batch_papers)}")

                if last_paper_index >= 0 and len(completed_papers) > 0:
                    print(f"[RESUME] Re-evaluating paper {last_paper_index + 1}: {pause_state.paper_doi}")

                    last_paper_result = completed_papers[-1]
                    paper_dir = Path(last_paper_result["paper_dir"])

                    print(f"  Re-running evaluation with potentially updated cache...")
                    evaluation_results = self._re_evaluate_paper(
                        paper_dir=paper_dir,
                        doi=last_paper_result["doi"]
                    )

                    completed_papers[-1]["metrics"] = self._extract_metrics_from_evaluation(
                        evaluation_results, last_paper_result["doi"]
                    )

                    print(f"  ✓ Re-evaluation complete")
                    print(f"  Updated metrics: P1_prec={completed_papers[-1]['metrics']['part1_precision']:.3f}, "
                          f"P1_rec={completed_papers[-1]['metrics']['part1_recall']:.3f}")
                else:
                    print(f"[RESUME] No completed papers found (crash during first paper processing)")
                    print(f"[RESUME] Will restart from paper 1: {pause_state.paper_doi}")

                if self.verify_only and last_paper_index >= 0 and len(completed_papers) > 0:
                    print(f"\n{'='*80}")
                    print("[VERIFY-ONLY] Re-evaluation complete. Pipeline paused again.")
                    print(f"{'='*80}")
                    print(f"\n📊 Review the updated metrics above.")

                    cache_path = self.config["evaluation"].get("cache_file_path", "")
                    if cache_path:
                        print(f"\n🔍 Semantic cache: {cache_path}")

                    print(f"\n📝 Next steps:")
                    print(f"  ① If semantic cache needs more corrections:")
                    print(f"     - Edit: {cache_path}")
                    print(f"     - Run: --resume --verify-only (to re-evaluate again)")
                    print(f"\n  ② If satisfied with the evaluation results:")
                    print(f"     - Run: --resume (without --verify-only to continue to next paper)")
                    print(f"\n{'='*80}\n")

                    batch_papers = pause_state.batch_info["batch_papers"]

                    completed_papers_dict = {p["doi"]: p for p in completed_papers}
                    aggregate_metrics = self._calculate_aggregate_metrics(completed_papers_dict)

                    raise PipelinePausedException(
                        f"Pipeline paused in verify-only mode for paper {last_paper_index + 1}/{len(batch_papers)}.",
                        round_num=round_num,
                        round_dir=str(round_dir),
                        aggregate_metrics=aggregate_metrics
                    )

                print(f"\n[RESUME] Continuing to next paper...")

                start_index = last_paper_index + 1
                paper_results = {p["doi"]: p for p in completed_papers}

            else:
                print(f"\n[RESUME] Cross-round resume detected:")
                print(f"  Pause state from Round {pause_round}, but current round is {round_num}")
                print(f"  Clearing old pause state and starting Round {round_num} fresh")

                pause_manager.clear_pause_state()

                start_index = 0
                paper_results = {}

        else:
            start_index = 0
            paper_results = {}

        for idx in range(start_index, len(batch_papers)):
            paper_info = batch_papers[idx]
            doi = paper_info["doi"]
            pmc_id = paper_info.get("pmc_id")

            print(f"\n{'-'*80}")
            print(f"Processing Paper {idx + 1}/{len(batch_papers)}")
            print(f"DOI: {doi}")
            if pmc_id:
                print(f"PMC ID: {pmc_id}")
            print(f"{'-'*80}")

            cache_path = self.config["evaluation"].get("cache_file_path", "")
            completed_papers_list = [paper_results[p["doi"]] for p in batch_papers[:idx] if p["doi"] in paper_results]

            pre_process_pause_state = create_pause_state_for_batch(
                paper_doi=doi,
                round=round_num,
                paper_dir="",
                round_dir=str(round_dir),
                config_path=str(self.config_path),
                cache_file_path=cache_path,
                paper_index_in_batch=idx - 1 if idx > 0 else -1,
                completed_papers=completed_papers_list,
                batch_papers=[{"doi": p["doi"], "pmc_id": p.get("pmc_id")} for p in batch_papers],
                prompt_version=self.prompt_version
            )
            pause_manager.save_pause_state(pre_process_pause_state)

            try:
                paper_result = self._extract_paper(
                    doi=doi,
                    pmc_id=pmc_id,
                    output_dir=batch_results_dir,
                    round_dir=round_dir,
                )
                paper_results[doi] = paper_result

                print(f"✓ Paper {idx + 1} processed successfully")
                print(f"  Metrics: P1_prec={paper_result['metrics']['part1_precision']:.3f}, "
                      f"P1_rec={paper_result['metrics']['part1_recall']:.3f}, "
                      f"P2_prec={paper_result['metrics']['part2_precision']:.3f}, "
                      f"P2_rec={paper_result['metrics']['part2_recall']:.3f}")

                cache_path = self.config["evaluation"].get("cache_file_path", "")
                completed_papers_list = [paper_results[p["doi"]] for p in batch_papers[:idx+1] if p["doi"] in paper_results]

                paper_pause_state = create_pause_state_for_batch(
                    paper_doi=doi,
                    round=round_num,
                    paper_dir=str(paper_result.get("paper_dir", "")),
                    round_dir=str(round_dir),
                    config_path=str(self.config_path),
                    cache_file_path=cache_path,
                    paper_index_in_batch=idx,
                    completed_papers=completed_papers_list,
                    batch_papers=[{"doi": p["doi"], "pmc_id": p.get("pmc_id")} for p in batch_papers],
                    prompt_version=self.prompt_version
                )

                paper_dir_path = Path(paper_result.get("paper_dir", ""))
                if paper_dir_path.exists():
                    paper_pause_state_path = paper_dir_path / "pause_state.json"
                    with open(paper_pause_state_path, "w", encoding="utf-8") as f:
                        json.dump(paper_pause_state.to_dict(), f, indent=2, ensure_ascii=False)
                    print(f"  Saved pause_state to: {paper_pause_state_path}")

            except Exception as e:
                print(f"✗ Error processing paper {idx + 1} ({doi}): {e}")
                paper_results[doi] = {
                    "status": "error",
                    "error": str(e),
                    "doi": doi,
                    "pmc_id": pmc_id,
                    "metrics": {
                        "part1_precision": 0.0,
                        "part1_recall": 0.0,
                        "part2_precision": 0.0,
                        "part2_recall": 0.0,
                    }
                }

            if self.pause_after_extraction:
                print(f"\n[PAUSE] Pausing after paper {idx + 1}/{len(batch_papers)}...")

                aggregate_metrics = self._calculate_aggregate_metrics(paper_results)

                aggregate_path = round_dir / "aggregate_metrics.json"
                with open(aggregate_path, "w", encoding="utf-8") as f:
                    json.dump(aggregate_metrics, f, indent=2, ensure_ascii=False)
                print(f"[PAUSE] Saved aggregate metrics: {aggregate_path}")

                completed_papers_list = [paper_results[p["doi"]] for p in batch_papers[:idx+1] if p["doi"] in paper_results]

                cache_path = self.config["evaluation"].get("cache_file_path", "")

                current_paper_result = paper_results.get(doi, {})
                pause_state = create_pause_state_for_batch(
                    paper_doi=doi,
                    round=round_num,
                    paper_dir=str(current_paper_result.get("paper_dir", "")),
                    round_dir=str(round_dir),
                    config_path=str(self.config_path),
                    cache_file_path=cache_path,
                    paper_index_in_batch=idx,
                    completed_papers=completed_papers_list,
                    batch_papers=[{"doi": p["doi"], "pmc_id": p.get("pmc_id")} for p in batch_papers],
                    prompt_version=self.prompt_version
                )

                pause_manager.save_pause_state(pause_state)

                print(f"[PAUSE] Semantic cache location: {cache_path}")
                print(f"[PAUSE] You can now review the cache and use --resume to continue")

                raise PipelinePausedException(
                    f"Pipeline paused after paper {idx + 1}/{len(batch_papers)}. "
                    f"Use --resume to continue.",
                    round_num=round_num,
                    round_dir=str(round_dir),
                    aggregate_metrics=aggregate_metrics
                )

        print(f"\n{'='*80}")
        print("All Papers Processed - Calculating Aggregate Metrics")
        print(f"{'='*80}")

        aggregate_metrics = self._calculate_aggregate_metrics(paper_results)
        batch_status = self._check_batch_acceptance(aggregate_metrics)

        if batch_status["status"] == "reject":
            self._run_batch_error_analysis(
                paper_results=paper_results,
                round_dir=round_dir,
                aggregate_metrics=aggregate_metrics
            )

            update_success = self._update_prompt_from_batch_errors(
                round_dir=round_dir,
                aggregate_metrics=aggregate_metrics
            )

            if update_success:
                print(f"✓ Prompt updated for next round")

                if self.pause_after_extraction:
                    self._save_round_results(
                        round_dir=round_dir,
                        paper_results=paper_results,
                        aggregate_metrics=aggregate_metrics,
                        batch_status=batch_status,
                        round_num=round_num,
                        timestamp=timestamp,
                    )

                    pause_state = PauseState(
                        paper_doi="",
                        round=round_num,
                        results_dir=str(self.experiment_dir),
                        paper_dir="",
                        round_dir=str(round_dir),
                        config_path=str(self.config_path),
                        input_prompt_path=str(self.experiment_dir / f"{self.prompt_version:02d}_prompt.md"),
                        cache_file_path="",
                        phase="prompt_updated",
                        paper_index=-1,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        batch_info={
                            "prompt_version": self.prompt_version,
                            "next_round": round_num + 1,
                        }
                    )
                    pause_manager.save_pause_state(pause_state)

                    print(f"\n{'='*80}")
                    print(f"[PAUSE] Pipeline paused after prompt update")
                    print(f"[PAUSE] Updated prompt: {self.experiment_dir / f'{self.prompt_version:02d}_prompt.md'}")
                    print(f"[PAUSE] Use --resume to start Round {round_num + 1} with the updated prompt")
                    print(f"{'='*80}\n")

                    raise PipelinePausedException(
                        f"Pipeline paused after prompt update. Use --resume to start Round {round_num + 1}.",
                        round_num=round_num,
                        round_dir=str(round_dir),
                        aggregate_metrics=aggregate_metrics
                    )
            else:
                print(f"⚠ Prompt update failed or skipped")

        self._save_round_results(
            round_dir=round_dir,
            paper_results=paper_results,
            aggregate_metrics=aggregate_metrics,
            batch_status=batch_status,
            round_num=round_num,
            timestamp=timestamp,
        )

        print(f"\n{'='*80}")
        print(f"Round {round_num} Complete")
        print(f"{'='*80}")
        print(f"Status: {batch_status['status'].upper()}")
        print(f"Results: {round_dir}")
        print(f"{'='*80}\n")

        pause_manager.clear_pause_state()

        return {
            "round_num": round_num,
            "round_dir": str(round_dir),
            "paper_results": paper_results,
            "aggregate_metrics": aggregate_metrics,
            "batch_status": batch_status,
        }

    def _extract_paper(
        self,
        doi: str,
        pmc_id: Optional[str],
        output_dir: Path,
        round_dir: Path,
    ) -> Dict[str, Any]:

        safe_doi = doi.replace("/", "_").replace(":", "_")
        paper_dir = output_dir / f"paper_{safe_doi}"
        paper_dir.mkdir(parents=True, exist_ok=True)

        print("  Running extraction pipeline (3 rounds)...")

        temp_config = self.config.copy()
        temp_config["results"]["results_dir"] = str(paper_dir)

        filter_data = [{"doi": doi, "pmc_id": pmc_id}] if pmc_id else [{"doi": doi, "pmc_id": None}]
        filter_file = paper_dir / "temp_paper_filter.json"
        with open(filter_file, "w", encoding="utf-8") as f:
            json.dump(filter_data, f)
        temp_config["data"]["paper_filter_file"] = str(filter_file)

        print("  Cleaning paper on-demand...")
        clean_json_papers_batch(temp_config)

        run_llm_pipeline(
            config=temp_config,
            read_papers=True,
            md_prompt=self.current_prompt,
            with_history=self.with_history,
            no_timestamp=True,
            custom_output_dir=paper_dir,
        )

        print("  Generating CSV...")
        save_csv(
            config_file=temp_config,
            results_dir=str(paper_dir),
        )

        print("  Processing concentration units...")
        csv_output_path = paper_dir / "data.csv"
        if csv_output_path.exists():
            processed_csv_path = process_units(
                csv_file_path=str(csv_output_path),
            )
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_output_path}")

        print("  Creating filtered labeled data for evaluation...")
        filtered_labeled_path = self._create_filtered_labeled_data(
            doi=doi,
            output_dir=paper_dir
        )

        print("  Running evaluation...")
        evaluation_results = run_evaluation(
            config_file=temp_config,
            pred_csv_path=processed_csv_path,
            labeled_csv_path=filtered_labeled_path,
        )

        if evaluation_results is None:
            raise ValueError("Evaluation failed - no results returned")

        part1_record_level = evaluation_results.get("part1_record_level", {})
        part2_entry_level = evaluation_results.get("part2_entry_level", {})

        per_doi_metrics = part1_record_level.get("per_doi_metrics", {})
        per_doi_field_metrics = part2_entry_level.get("per_doi_field_metrics", {})

        if doi in per_doi_metrics:
            part1_precision = per_doi_metrics[doi]["precision"]
            part1_recall = per_doi_metrics[doi]["recall"]
        else:
            if per_doi_metrics:
                first_doi = list(per_doi_metrics.keys())[0]
                part1_precision = per_doi_metrics[first_doi]["precision"]
                part1_recall = per_doi_metrics[first_doi]["recall"]
            else:
                part1_precision = 0.0
                part1_recall = 0.0

        if doi in per_doi_field_metrics:
            part2_precision = per_doi_field_metrics[doi]["precision"]
            part2_recall = per_doi_field_metrics[doi]["recall"]
        else:
            if per_doi_field_metrics:
                first_doi = list(per_doi_field_metrics.keys())[0]
                part2_precision = per_doi_field_metrics[first_doi]["precision"]
                part2_recall = per_doi_field_metrics[first_doi]["recall"]
            else:
                part2_precision = 0.0
                part2_recall = 0.0

        metrics = {
            "part1_precision": part1_precision,
            "part1_recall": part1_recall,
            "part2_precision": part2_precision,
            "part2_recall": part2_recall,
        }

        if filter_file.exists():
            filter_file.unlink()

        return {
            "doi": doi,
            "pmc_id": pmc_id,
            "paper_dir": str(paper_dir),
            "clean_json_dir": str(temp_config["data"]["clean_json_dir"]),
            "metrics": metrics,
            "status": "completed",
        }

    def _re_evaluate_paper(
        self,
        paper_dir: Path,
        doi: str
    ) -> Dict:

        print(f"  Re-evaluating paper: {doi}")

        processed_csv_path = paper_dir / "data_processed.csv"

        if not processed_csv_path.exists():
            raise FileNotFoundError(f"Processed CSV not found: {processed_csv_path}")

        csv_path = paper_dir / "data.csv"
        if csv_path.exists():
            print(f"  Re-running process_units() to regenerate data_processed.csv with latest logic...")
            try:
                from tpd_curator.pipeline import process_units
                new_processed_csv_path = process_units(csv_file_path=str(csv_path))
                processed_csv_path = Path(new_processed_csv_path)
                print(f"  Regenerated: {processed_csv_path}")
            except Exception as e:
                print(f"  Warning: Could not re-run process_units(): {e}")
                print(f"  Using existing processed CSV: {processed_csv_path}")
        else:
            print(f"  Using existing processed CSV: {processed_csv_path}")

        temp_config = self.config.copy()
        temp_config["results"]["results_dir"] = str(paper_dir)

        print(f"  Creating filtered labeled data for re-evaluation...")
        filtered_labeled_path = self._create_filtered_labeled_data(
            doi=doi,
            output_dir=paper_dir
        )

        evaluation_results = run_evaluation(
            config_file=temp_config,
            pred_csv_path=str(processed_csv_path),
            labeled_csv_path=filtered_labeled_path,
        )

        if evaluation_results is None:
            raise ValueError("Evaluation failed")

        return evaluation_results

    def _extract_metrics_from_evaluation(
        self,
        evaluation_results: Dict,
        doi: str
    ) -> Dict[str, float]:

        part1_record_level = evaluation_results.get("part1_record_level", {})
        part2_entry_level = evaluation_results.get("part2_entry_level", {})

        per_doi_metrics = part1_record_level.get("per_doi_metrics", {})
        per_doi_field_metrics = part2_entry_level.get("per_doi_field_metrics", {})

        if doi in per_doi_metrics:
            part1_precision = per_doi_metrics[doi]["precision"]
            part1_recall = per_doi_metrics[doi]["recall"]
        else:
            part1_precision = 0.0
            part1_recall = 0.0

        if doi in per_doi_field_metrics:
            part2_precision = per_doi_field_metrics[doi]["precision"]
            part2_recall = per_doi_field_metrics[doi]["recall"]
        else:
            part2_precision = 0.0
            part2_recall = 0.0

        return {
            "part1_precision": part1_precision,
            "part1_recall": part1_recall,
            "part2_precision": part2_precision,
            "part2_recall": part2_recall,
        }

    def _calculate_aggregate_metrics(
        self,
        paper_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:

        valid_results = {
            doi: result for doi, result in paper_results.items()
            if result["status"] == "completed"
        }

        if not valid_results:
            raise ValueError("No valid paper results to calculate aggregate metrics")

        num_papers = len(valid_results)

        avg_part1_precision = sum(
            r["metrics"]["part1_precision"] for r in valid_results.values()
        ) / num_papers

        avg_part1_recall = sum(
            r["metrics"]["part1_recall"] for r in valid_results.values()
        ) / num_papers

        avg_part2_precision = sum(
            r["metrics"]["part2_precision"] for r in valid_results.values()
        ) / num_papers

        avg_part2_recall = sum(
            r["metrics"]["part2_recall"] for r in valid_results.values()
        ) / num_papers

        per_paper_metrics = {
            doi: result["metrics"]
            for doi, result in valid_results.items()
        }

        aggregate = {
            "avg_part1_precision": avg_part1_precision,
            "avg_part1_recall": avg_part1_recall,
            "avg_part2_precision": avg_part2_precision,
            "avg_part2_recall": avg_part2_recall,
            "precision_threshold": self.precision_threshold,
            "recall_threshold": self.recall_threshold,
            "num_papers": num_papers,
            "per_paper_metrics": per_paper_metrics,
        }

        print(f"\nAggregate Metrics (Average of {num_papers} papers):")
        print(f"  Part 1 Precision: {avg_part1_precision:.4f} (threshold: {self.precision_threshold})")
        print(f"  Part 1 Recall:    {avg_part1_recall:.4f} (threshold: {self.recall_threshold})")
        print(f"  Part 2 Precision: {avg_part2_precision:.4f} (threshold: {self.precision_threshold})")
        print(f"  Part 2 Recall:    {avg_part2_recall:.4f} (threshold: {self.recall_threshold})")

        return aggregate

    def _check_batch_acceptance(
        self,
        aggregate_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:

        failing_metrics = []

        if aggregate_metrics["avg_part1_precision"] < self.precision_threshold:
            failing_metrics.append("avg_part1_precision")

        if aggregate_metrics["avg_part1_recall"] < self.recall_threshold:
            failing_metrics.append("avg_part1_recall")

        if aggregate_metrics["avg_part2_precision"] < self.precision_threshold:
            failing_metrics.append("avg_part2_precision")

        if aggregate_metrics["avg_part2_recall"] < self.recall_threshold:
            failing_metrics.append("avg_part2_recall")

        status = "accept" if not failing_metrics else "reject"

        batch_status = {
            "status": status,
            "failing_metrics": failing_metrics,
        }

        print(f"\nBatch Status: {status.upper()}")
        if failing_metrics:
            print(f"Failing Metrics: {', '.join(failing_metrics)}")
        else:
            print("All metrics meet thresholds!")

        return batch_status

    def _save_round_results(
        self,
        round_dir: Path,
        paper_results: Dict[str, Dict[str, Any]],
        aggregate_metrics: Dict[str, Any],
        batch_status: Dict[str, Any],
        round_num: int,
        timestamp: str,
    ):

        aggregate_path = round_dir / "aggregate_metrics.json"
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nSaved aggregate metrics: {aggregate_path}")

        status_data = {
            "status": batch_status["status"],
            "failing_metrics": batch_status["failing_metrics"],
            "round": round_num,
            "timestamp": timestamp,
        }
        status_path = round_dir / "batch_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
        print(f"Saved batch status: {status_path}")

        print(f"\nAll results saved to: {round_dir}")


    def _get_paper_main_text(self, pmc_id: str, clean_json_dir: Optional[str] = None) -> str:

        if clean_json_dir:
            json_file_dir = Path(clean_json_dir)
        else:
            json_file_dir = Path(self.config["data"]["clean_json_dir"])

        json_file = json_file_dir / f"{pmc_id}_cleaned.json"

        if not json_file.exists():
            print(f"  Warning: Cleaned JSON not found: {json_file}")
            return "[Paper content not available]"

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                paper_data = json.load(f)

            paper_text = paper_data.get("full_text", "")

            if not paper_text:
                abstract = paper_data.get("abstract", "")
                sections = paper_data.get("sections", {})
                if sections:
                    section_text = "\n\n".join([
                        f"{heading}\n{text}"
                        for heading, text in sections.items()
                    ])
                    paper_text = f"{abstract}\n\n{section_text}"
                else:
                    paper_text = abstract

            return paper_text

        except Exception as e:
            print(f"  Error reading cleaned JSON: {e}")
            return "[Error loading paper content]"

    def _create_filtered_labeled_data(
        self,
        doi: str,
        output_dir: Path
    ) -> str:

        labeled_data_path = self.config["evaluation"]["labeled_data_path"]

        try:
            df = pd.read_csv(labeled_data_path)

            df_filtered = df[df["DOI"] == doi]

            if len(df_filtered) == 0:
                print(f"  Warning: No labeled data found for DOI: {doi}")

            filtered_path = output_dir / "filtered_labeled_data.csv"
            df_filtered.to_csv(filtered_path, index=False, encoding="utf-8")

            print(f"  Created filtered labeled data: {len(df_filtered)} records for {doi}")

            return str(filtered_path)

        except Exception as e:
            print(f"  Error creating filtered labeled data: {e}")
            return labeled_data_path

    def _load_ground_truth_for_doi(self, doi: str) -> List[Dict]:

        labeled_data_path = self.config["evaluation"]["labeled_data_path"]

        try:
            df = pd.read_csv(labeled_data_path)

            df_doi = df[df["DOI"] == doi]

            if len(df_doi) == 0:
                print(f"  Warning: No ground truth found for DOI: {doi}")

            return df_doi.to_dict('records')

        except Exception as e:
            print(f"  Error loading ground truth: {e}")
            return []

    def _add_evaluation_labels_to_predictions(
        self,
        predictions: List[Dict],
        doi: str,
        detailed_evaluation: Dict
    ) -> List[Dict]:

        if not detailed_evaluation or str(doi) not in detailed_evaluation:
            for pred in predictions:
                pred['evaluation_label'] = 'unknown'
            return predictions

        doi_eval = detailed_evaluation[str(doi)]['part1_record_level']
        tp_records = doi_eval.get('tp_records', [])
        fp_records = doi_eval.get('fp_records', [])

        for pred in predictions:
            pred_compound = str(pred.get('Compound_Name', '')).strip() if pd.notna(pred.get('Compound_Name')) else ''
            pred_target = str(pred.get('Degradation_Target', '')).strip() if pd.notna(pred.get('Degradation_Target')) else ''
            pred_recruiter = str(pred.get('Recruiter', '')).strip() if pd.notna(pred.get('Recruiter')) else ''
            pred_assay = str(pred.get('Assay', '')).strip() if pd.notna(pred.get('Assay')) else ''
            pred_cell = str(pred.get('Cell_Line', '')).strip() if pd.notna(pred.get('Cell_Line')) else ''

            matched_tp = False
            for tp in tp_records:
                tp_compound = str(tp.get('pred_compound', '')).strip() if pd.notna(tp.get('pred_compound')) else ''
                tp_target = str(tp.get('pred_target', '')).strip() if pd.notna(tp.get('pred_target')) else ''
                tp_recruiter = str(tp.get('pred_recruiter', '')).strip() if pd.notna(tp.get('pred_recruiter')) else ''
                tp_assay = str(tp.get('pred_assay', '')).strip() if pd.notna(tp.get('pred_assay')) else ''
                tp_cell = str(tp.get('pred_cell_line', '')).strip() if pd.notna(tp.get('pred_cell_line')) else ''

                if (tp_compound.lower() == pred_compound.lower() and
                    tp_target.lower() == pred_target.lower() and
                    tp_recruiter.lower() == pred_recruiter.lower() and
                    tp_assay.lower() == pred_assay.lower() and
                    tp_cell.lower() == pred_cell.lower()):
                    pred['evaluation_label'] = 'TP'
                    pred['match_reason'] = tp.get('match_reason', '')
                    matched_tp = True
                    break

            if matched_tp:
                continue

            matched_fp = False
            for fp in fp_records:
                fp_compound = str(fp.get('pred_compound', '')).strip() if pd.notna(fp.get('pred_compound')) else ''
                fp_target = str(fp.get('pred_target', '')).strip() if pd.notna(fp.get('pred_target')) else ''
                fp_recruiter = str(fp.get('pred_recruiter', '')).strip() if pd.notna(fp.get('pred_recruiter')) else ''
                fp_assay = str(fp.get('pred_assay', '')).strip() if pd.notna(fp.get('pred_assay')) else ''
                fp_cell = str(fp.get('pred_cell_line', '')).strip() if pd.notna(fp.get('pred_cell_line')) else ''

                if (fp_compound.lower() == pred_compound.lower() and
                    fp_target.lower() == pred_target.lower() and
                    fp_recruiter.lower() == pred_recruiter.lower() and
                    fp_assay.lower() == pred_assay.lower() and
                    fp_cell.lower() == pred_cell.lower()):
                    pred['evaluation_label'] = 'FP'
                    matched_fp = True
                    break

            if not matched_fp:
                pred['evaluation_label'] = 'unknown'

        for pred in predictions:
            if pred.get('evaluation_label') == 'TP':
                pred.pop('evaluation_label', None)
                pred.pop('match_reason', None)

                pred['Compound_Name_Evaluation'] = 'TP'
                pred['Degradation_Target_Evaluation'] = 'TP'
                pred['Recruiter_Evaluation'] = 'TP'
                pred['Assay_Evaluation'] = 'TP'
                pred['Cell_Line_Evaluation'] = 'TP'

                if str(doi) in detailed_evaluation:
                    doi_eval = detailed_evaluation[str(doi)]
                    part2 = doi_eval.get('part2_field_level', {})
                    field_level_data = part2.get('matched_records', [])

                    if field_level_data:
                        matched_field_record = self._match_field_level_evaluation(
                            prediction=pred,
                            field_level_data=field_level_data,
                            doi=doi
                        )

                        if matched_field_record:
                            self._add_field_level_labels(pred, matched_field_record)
                        else:
                            compound = pred.get('Compound_Name', 'unknown')
                            print(f"    Info: No field-level match found for TP record: {compound}")

        return predictions

    def _match_field_level_evaluation(
        self,
        prediction: Dict,
        field_level_data: List[Dict],
        doi: str
    ) -> Optional[Dict]:

        pred_compound = str(prediction.get('Compound_Name', '')).strip() if pd.notna(prediction.get('Compound_Name')) else ''

        if not pred_compound:
            return None

        pred_compound_lower = pred_compound.lower()
        candidates = [
            rec for rec in field_level_data
            if str(rec.get('compound', '')).strip().lower() == pred_compound_lower
        ]

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        pred_dc50 = prediction.get('DC50', '')

        if pd.notna(pred_dc50) and str(pred_dc50).strip():
            pred_dc50_str = str(pred_dc50).strip()

            for candidate in candidates:
                fields = candidate.get('fields', {})
                dc50_field = fields.get('DC50', {})
                candidate_dc50 = dc50_field.get('pred_value', '')

                if pd.notna(candidate_dc50) and str(candidate_dc50).strip():
                    candidate_dc50_str = str(candidate_dc50).strip()

                    try:
                        if float(pred_dc50_str) == float(candidate_dc50_str):
                            return candidate
                    except (ValueError, TypeError):
                        if pred_dc50_str == candidate_dc50_str:
                            return candidate

        print(f"    Warning: Multiple field-level matches for '{pred_compound}' in {doi}, using first match (DC50 comparison inconclusive)")
        return candidates[0]

    def _add_field_level_labels(
        self,
        prediction: Dict,
        field_level_record: Dict
    ) -> Dict:

        evaluated_fields = ['DC50', 'DC50_units', 'DC50_h', 'Dmax', 'Dmax_h', 'Dmax_conc']

        fields_data = field_level_record.get('fields', {})

        for field_name in evaluated_fields:
            if field_name in fields_data:
                field_eval = fields_data[field_name]

                status = field_eval.get('status', 'unknown')
                prediction[f'{field_name}_Evaluation'] = status

                reason = field_eval.get('reason', '')
                if reason:
                    if status in ['FP', 'FN']:
                        prediction[f'{field_name}_Evaluation_Reason'] = reason
                    elif reason not in ['Both empty', 'Values matched']:
                        prediction[f'{field_name}_Evaluation_Reason'] = reason

        return prediction

    def _find_ground_truth_by_keys(
        self,
        error_record: Dict,
        ground_truth_list: List[Dict],
        key_fields: List[str]
    ) -> Optional[Dict]:

        field_mapping = {
            'label_compound': 'Compound_Name',
            'label_target': 'Degradation_Target',
            'label_recruiter': 'Recruiter',
            'label_assay': 'Assay',
            'label_cell_line': 'Cell_Line'
        }

        for gt in ground_truth_list:
            all_match = True
            for label_field in key_fields:
                gt_field = field_mapping.get(label_field, label_field)

                error_val = str(error_record.get(label_field, '')).strip().lower()
                gt_val = str(gt.get(gt_field, '')).strip().lower()

                if error_val != gt_val:
                    all_match = False
                    break

            if all_match:
                return gt

        return None

    def _find_prediction_by_keys(
        self,
        error_record: Dict,
        predictions_list: List[Dict],
        key_fields: List[str]
    ) -> Optional[Dict]:

        field_mapping = {
            'pred_compound': 'Compound_Name',
            'pred_target': 'Degradation_Target',
            'pred_recruiter': 'Recruiter',
            'pred_assay': 'Assay',
            'pred_cell_line': 'Cell_Line'
        }

        for pred in predictions_list:
            all_match = True
            for pred_field in key_fields:
                pred_csv_field = field_mapping.get(pred_field, pred_field)

                error_val = str(error_record.get(pred_field, '')).strip().lower()
                pred_val = str(pred.get(pred_csv_field, '')).strip().lower()

                if error_val != pred_val:
                    all_match = False
                    break

            if all_match:
                return pred

        return None

    def _find_ground_truth_by_compound(
        self,
        compound_name: str,
        ground_truth_list: List[Dict]
    ) -> Optional[Dict]:

        compound_lower = compound_name.strip().lower()

        for gt in ground_truth_list:
            gt_compound = str(gt.get('Compound_Name', '')).strip().lower()
            if gt_compound == compound_lower:
                return gt

        return None

    def _find_prediction_by_compound(
        self,
        compound_name: str,
        predictions_list: List[Dict]
    ) -> Optional[Dict]:

        compound_lower = compound_name.strip().lower()

        for pred in predictions_list:
            pred_compound = str(pred.get('Compound_Name', '')).strip().lower()
            if pred_compound == compound_lower:
                return pred

        return None

    def _classify_errors_by_type(
        self,
        doi: str,
        detailed_evaluation: Dict,
        all_ground_truth: List[Dict],
        all_predictions: List[Dict]
    ) -> Dict[str, Any]:

        doi_eval = detailed_evaluation.get(str(doi), {})
        part1 = doi_eval.get('part1_record_level', {})
        part2 = doi_eval.get('part2_field_level', {})

        fn_records = part1.get('fn_records', [])
        group1_errors = []

        for fn in fn_records:
            gt_record = self._find_ground_truth_by_keys(
                fn,
                all_ground_truth,
                key_fields=['label_compound', 'label_target', 'label_recruiter',
                           'label_assay', 'label_cell_line']
            )

            if gt_record:
                group1_errors.append({
                    'fn_record': fn,
                    'gt_full_record': gt_record
                })
            else:
                print(f"    Warning: Could not find full GT record for FN: {fn.get('label_compound', 'unknown')}")
                group1_errors.append({
                    'fn_record': fn,
                    'gt_full_record': {}
                })

        group2_errors = []

        fp_records = part1.get('fp_records', [])

        for fp in fp_records:
            pred_record = self._find_prediction_by_keys(
                fp,
                all_predictions,
                key_fields=['pred_compound', 'pred_target', 'pred_recruiter',
                           'pred_assay', 'pred_cell_line']
            )

            if pred_record:
                fp_reason = fp.get('fp_reason', 'Unknown FP reason')
                fp_summary = fp.get('fp_summary', {})

                reason_parts = [fp_reason]
                if fp_summary and 'quick_explanation' in fp_summary:
                    reason_parts.append(fp_summary['quick_explanation'])

                group2_errors.append({
                    'error_type': 'record_level_fp',
                    'extracted': {
                        'Compound_Name': pred_record.get('Compound_Name', ''),
                        'Degradation_Target': pred_record.get('Degradation_Target', ''),
                        'Recruiter': pred_record.get('Recruiter', ''),
                        'Assay': pred_record.get('Assay', ''),
                        'Cell_Line': pred_record.get('Cell_Line', ''),
                        'DC50': pred_record.get('DC50', ''),
                        'DC50_units': pred_record.get('DC50_units', ''),
                        'DC50_h': pred_record.get('DC50_h', ''),
                        'Dmax': pred_record.get('Dmax', ''),
                        'Dmax_h': pred_record.get('Dmax_h', ''),
                        'Dmax_conc': pred_record.get('Dmax_conc', '')
                    },
                    'ground_truth': None,
                    'reason': ' - '.join(reason_parts)[:500]
                })
            else:
                print(f"    Warning: Could not find full prediction for FP: {fp.get('pred_compound', 'unknown')}")

        matched_records = part2.get('matched_records', [])
        field_keys = ['DC50', 'DC50_units', 'DC50_h', 'Dmax', 'Dmax_h', 'Dmax_conc']

        for matched_rec in matched_records:
            fields = matched_rec.get('fields', {})

            has_error = False
            error_field_names = []

            for field_name in field_keys:
                if field_name in fields:
                    field_data = fields[field_name]
                    status = field_data.get('status', '')

                    if status in ['FP', 'FN']:
                        has_error = True
                        error_field_names.append(f"{field_name} ({status})")

            if has_error:
                compound_name = matched_rec.get('compound', '')

                gt_record = self._find_ground_truth_by_compound(
                    compound_name, all_ground_truth
                )
                pred_record = self._find_prediction_by_compound(
                    compound_name, all_predictions
                )

                if not gt_record or not pred_record:
                    print(f"    Warning: Missing full record for field mismatch on compound '{compound_name}'")
                    continue

                reason = f"Record matched on 5-key fields (Compound, Target, Recruiter, Assay, Cell Line) but has errors in measurement fields: {', '.join(error_field_names)}"

                group2_errors.append({
                    'error_type': 'field_level_mismatch',
                    'extracted': {
                        'Compound_Name': pred_record.get('Compound_Name', ''),
                        'Degradation_Target': pred_record.get('Degradation_Target', ''),
                        'Recruiter': pred_record.get('Recruiter', ''),
                        'Assay': pred_record.get('Assay', ''),
                        'Cell_Line': pred_record.get('Cell_Line', ''),
                        'DC50': pred_record.get('DC50', ''),
                        'DC50_units': pred_record.get('DC50_units', ''),
                        'DC50_h': pred_record.get('DC50_h', ''),
                        'Dmax': pred_record.get('Dmax', ''),
                        'Dmax_h': pred_record.get('Dmax_h', ''),
                        'Dmax_conc': pred_record.get('Dmax_conc', '')
                    },
                    'ground_truth': {
                        'Compound_Name': gt_record.get('Compound_Name', ''),
                        'Degradation_Target': gt_record.get('Degradation_Target', ''),
                        'Recruiter': gt_record.get('Recruiter', ''),
                        'Assay': gt_record.get('Assay', ''),
                        'Cell_Line': gt_record.get('Cell_Line', ''),
                        'DC50': gt_record.get('DC50', ''),
                        'DC50_units': gt_record.get('DC50_units', ''),
                        'DC50_h': gt_record.get('DC50_h', ''),
                        'Dmax': gt_record.get('Dmax', ''),
                        'Dmax_h': gt_record.get('Dmax_h', ''),
                        'Dmax_conc': gt_record.get('Dmax_conc', '')
                    },
                    'reason': reason
                })

        return {
            'group1_record_level_omissions': {
                'count': len(group1_errors),
                'description': 'Records in ground truth but completely missing from extraction',
                'errors': group1_errors
            },
            'group2_unmatched_extra_data': {
                'count': len(group2_errors),
                'description': 'Unmatched or extra records: includes pure hallucinations and records with correct 5-key but wrong field values',
                'errors': group2_errors
            }
        }

    def _build_error_analysis_prompt(
        self,
        doi: str,
        paper_main_text: str,
        classified_errors: Dict[str, Any],
        metrics: Dict[str, float],
        all_ground_truth: List[Dict],
        semantic_matches: List[Dict] = None
    ) -> str:

        group1 = classified_errors['group1_record_level_omissions']
        group2 = classified_errors['group2_unmatched_extra_data']

        prompt = f"""I'm extracting molecular glue degradation assay measurements from a molecular glue paper using an LLM, specifically focusing on DC50 and Dmax values. The paper is delimited by triple backticks (``` ... ```). There are some errors (FP and FN) in the extraction results, I need you to analyze these errors to help me improve future extraction performance.

Here is the main text of the paper:
```
{paper_main_text}
```
**EXTRACTION ERROR CLASSIFICATION**:
The extraction errors have been classified into 2 groups:
"""

        prompt += f"""
## GROUP 1: DATA OMISSION (False Negatives)
**Definition**: Data that exist in ground truth but were missing from the extraction. 
"""

        if group1['count'] > 0:
            import json
            prompt += "**Ground Truth Records that were MISSED**:\n"
            for idx, error_ctx in enumerate(group1['errors'], 1):
                gt_record = error_ctx.get('gt_full_record', {})

                gt_data = {
                    'Compound_Name': gt_record.get('Compound_Name', ''),
                    'Degradation_Target': gt_record.get('Degradation_Target', ''),
                    'Recruiter': gt_record.get('Recruiter', ''),
                    'Assay': gt_record.get('Assay', ''),
                    'Cell_Line': gt_record.get('Cell_Line', ''),
                    'DC50': gt_record.get('DC50', ''),
                    'DC50_units': gt_record.get('DC50_units', ''),
                    'DC50_h': gt_record.get('DC50_h', ''),
                    'Dmax': gt_record.get('Dmax', ''),
                    'Dmax_h': gt_record.get('Dmax_h', ''),
                    'Dmax_conc': gt_record.get('Dmax_conc', '')
                }

                prompt += f"\nMissed Record #{idx}:\n"
                prompt += "```json\n"
                prompt += json.dumps(gt_data, indent=2, ensure_ascii=False)
                prompt += "\n```\n"
        else:
            prompt += "No data omissions.\n"

        prompt += f"""

## GROUP 2: UNMATCHED / EXTRA DATA (False Positives)
**Definition**: Data that were extracted but do not match ground truth. This includes:
  - Data not in ground truth.
  - Data with correct record level match but wrong measurement field values
"""

        if group2['count'] > 0:
            import json
            prompt += "**Unmatched / Extra Data Details**:\n\n"

            for idx, error_ctx in enumerate(group2['errors'], 1):
                error_type = error_ctx.get('error_type', 'unknown')
                extracted = error_ctx.get('extracted', {})

                extracted_json = json.dumps(extracted, indent=4, ensure_ascii=False)

                prompt += f"Error #{idx} ({error_type}):\n"
                prompt += "```json\n"
                prompt += "{\n"
                prompt += f'  "extracted": {extracted_json}'

                if error_type == 'record_level_fp':
                    reason = error_ctx.get('reason', 'No reason provided')
                    reason_json = json.dumps(reason, ensure_ascii=False)
                    prompt += f',\n  "reason": {reason_json}'

                prompt += "\n}\n"
                prompt += "```\n\n"
        else:
            prompt += "No unmatched or extra data.\n"

        prompt += f"""

## GROUND TRUTH FOR THIS PAPER
**Reference**: The complete ground truth data for this paper:
"""
        if all_ground_truth:
            import json
            for idx, gt_record in enumerate(all_ground_truth, 1):
                gt_data = {
                    'Compound_Name': gt_record.get('Compound_Name', ''),
                    'Degradation_Target': gt_record.get('Degradation_Target', ''),
                    'Recruiter': gt_record.get('Recruiter', ''),
                    'Assay': gt_record.get('Assay', ''),
                    'Cell_Line': gt_record.get('Cell_Line', ''),
                    'DC50': gt_record.get('DC50', ''),
                    'DC50_units': gt_record.get('DC50_units', ''),
                    'DC50_h': gt_record.get('DC50_h', ''),
                    'Dmax': gt_record.get('Dmax', ''),
                    'Dmax_h': gt_record.get('Dmax_h', ''),
                    'Dmax_conc': gt_record.get('Dmax_conc', '')
                }
                prompt += f"\nRecord #{idx}:\n"
                prompt += "```json\n"
                prompt += json.dumps(gt_data, indent=2, ensure_ascii=False)
                prompt += "\n```\n"
        else:
            prompt += "No ground truth data available.\n"

        prompt += f"""

## SEMANTIC MATCHES FOR THIS PAPER
**Reference**: Semantic matching decisions made during evaluation (showing how predicted values were compared to ground truth values):
"""
        if semantic_matches:
            for idx, match in enumerate(semantic_matches, 1):
                pred_val = match.get('pred_value', 'N/A')
                label_val = match.get('labeled_value', 'N/A')
                feature = match.get('feature', 'N/A')
                result = match.get('result', 'N/A')
                prompt += f"- {feature}: '{pred_val}' vs '{label_val}' → {result}\n"
        else:
            prompt += "No semantic matches recorded for this paper.\n"

        prompt += """
---
**TASK**:
Accoding to above information, analyze the root causes of the errors. Summarize your findings in 2 sentences to improve the next extraction round. Let's think step by step.
"""

        return prompt

    def _analyze_paper_errors(
        self,
        doi: str,
        pmc_id: Optional[str],
        paper_dir: Path,
        metrics: Dict[str, float],
        clean_json_dir: Optional[str] = None
    ) -> str:

        print(f"  Analyzing errors for {doi}...")

        if not pmc_id:
            print("    Warning: No PMC ID, skipping paper content")
            paper_main_text = "[Paper content not available]"
        else:
            paper_main_text = self._get_paper_main_text(pmc_id, clean_json_dir=clean_json_dir)

        all_ground_truth = self._load_ground_truth_for_doi(doi)

        pred_csv = paper_dir / "data_processed.csv"
        if pred_csv.exists():
            df_pred = pd.read_csv(pred_csv)
            all_predictions = df_pred.to_dict('records')
        else:
            print("    Warning: No predictions CSV found")
            all_predictions = []

        eval_file = paper_dir / "record_based_evaluation_results.json"
        if eval_file.exists():
            evaluation = json.loads(eval_file.read_text())
        else:
            print("    Warning: No record_based_evaluation_results.json found")
            evaluation = {}

        detailed_eval_file = paper_dir / "evaluation_detailed_by_doi.json"
        detailed_evaluation = {}
        if detailed_eval_file.exists():
            try:
                with open(detailed_eval_file, "r", encoding="utf-8") as f:
                    detailed_evaluation = json.load(f)
                print("    ✓ Loaded detailed evaluation with per-record labels")
            except Exception as e:
                print(f"    Warning: Could not load detailed evaluation: {e}")
                detailed_evaluation = {}
        else:
            print("    Info: No detailed evaluation file found (using unlabeled predictions)")

        print("    Classifying errors by type...")
        classified_errors = self._classify_errors_by_type(
            doi=doi,
            detailed_evaluation=detailed_evaluation,
            all_ground_truth=all_ground_truth,
            all_predictions=all_predictions
        )

        group1_count = classified_errors['group1_record_level_omissions']['count']
        group2_count = classified_errors['group2_unmatched_extra_data']['count']
        print(f"    Group 1 (Data Omission - FN): {group1_count} errors")
        print(f"    Group 2 (Unmatched/Extra Data - FP): {group2_count} errors")

        semantic_matches = []
        cache_file_path = self.config.get("evaluation", {}).get("cache_file_path")
        if cache_file_path:
            from tpd_curator.pipeline import load_semantic_cache
            all_semantic_cache = load_semantic_cache(cache_file_path)
            semantic_matches = all_semantic_cache.get(str(doi), [])
            if "_legacy" in all_semantic_cache:
                semantic_matches = semantic_matches + all_semantic_cache.get("_legacy", [])
            print(f"    Loaded {len(semantic_matches)} semantic matches for this DOI")

        prompt = self._build_error_analysis_prompt(
            doi=doi,
            paper_main_text=paper_main_text,
            classified_errors=classified_errors,
            metrics=metrics,
            all_ground_truth=all_ground_truth,
            semantic_matches=semantic_matches
        )

        from tpd_curator.llm.run_llm_api import run_llm_api

        try:
            analysis_text, model_name = run_llm_api(
                custom_prompt=prompt,
                paper_text=None,
                temperature=0.1
            )

            if isinstance(analysis_text, (list, dict)):
                analysis_text = json.dumps(analysis_text, indent=2, ensure_ascii=False)

            if analysis_text.startswith("```"):
                lines = analysis_text.split("\n")
                analysis_text = "\n".join([
                    line for line in lines
                    if not line.startswith("```")
                ])

        except Exception as e:
            print(f"    Error calling LLM: {e}")
            analysis_text = f"Error during LLM analysis: {e}"

        error_analysis_file = paper_dir / "error_analysis.txt"
        error_analysis_file.write_text(analysis_text, encoding="utf-8")

        print(f"    ✓ Error analysis saved")

        return analysis_text

    def _run_batch_error_analysis(
        self,
        paper_results: Dict[str, Dict],
        round_dir: Path,
        aggregate_metrics: Dict[str, Any]
    ):

        print(f"\n{'='*80}")
        print("Running Error Analysis for Rejected Batch")
        print(f"{'='*80}\n")

        all_analyses = {}

        for doi, result in paper_results.items():
            if result["status"] != "completed":
                print(f"Skipping {doi} (status: {result['status']})")
                continue

            pmc_id = result.get("pmc_id")
            paper_dir = Path(result["paper_dir"])
            metrics = result["metrics"]
            clean_json_dir = result.get("clean_json_dir")

            all_metrics_meet_threshold = (
                metrics["part1_precision"] >= self.precision_threshold and
                metrics["part1_recall"] >= self.recall_threshold and
                metrics["part2_precision"] >= self.precision_threshold and
                metrics["part2_recall"] >= self.recall_threshold
            )

            if all_metrics_meet_threshold:
                print(f"Skipping {doi} (all metrics meet threshold: "
                      f"P1={metrics['part1_precision']:.3f}/{metrics['part1_recall']:.3f}, "
                      f"P2={metrics['part2_precision']:.3f}/{metrics['part2_recall']:.3f})")
                continue

            try:
                analysis = self._analyze_paper_errors(
                    doi=doi,
                    pmc_id=pmc_id,
                    paper_dir=paper_dir,
                    metrics=metrics,
                    clean_json_dir=clean_json_dir
                )
                all_analyses[doi] = {
                    "analysis": analysis,
                    "metrics": metrics,
                    "paper_dir": result["paper_dir"]
                }
            except Exception as e:
                print(f"  ✗ Error analyzing {doi}: {e}")
                import traceback
                traceback.print_exc()
                all_analyses[doi] = {
                    "analysis": f"Error during analysis: {e}",
                    "metrics": metrics,
                    "paper_dir": result["paper_dir"]
                }

        self._save_error_analysis_summary(
            all_analyses,
            round_dir,
            self.precision_threshold,
            self.recall_threshold,
            aggregate_metrics
        )

        print(f"\n✓ Error analysis complete for {len(all_analyses)} papers")

    def _save_error_analysis_summary(
        self,
        all_analyses: Dict[str, Dict],
        round_dir: Path,
        precision_threshold: float,
        recall_threshold: float,
        aggregate_metrics: Dict[str, Any]
    ):

        summary_lines = [
            "# Error Analysis Summary\n",
            f"\n**Paper Status**: REJECTED",
            f"**Number of Papers**: {len(all_analyses)}\n",
            "---\n"
        ]

        for paper_index, (doi, data) in enumerate(all_analyses.items(), start=1):
            metrics = data["metrics"]
            analysis = data["analysis"]
            paper_dir = Path(data["paper_dir"])

            eval_file = paper_dir / "record_based_evaluation_results.json"
            evaluation = {}
            if eval_file.exists():
                try:
                    with open(eval_file, "r", encoding="utf-8") as f:
                        evaluation = json.load(f)
                except Exception as e:
                    print(f"  Warning: Failed to load evaluation for Paper {paper_index}: {e}")
                    evaluation = {}
            else:
                print(f"  Warning: Evaluation file not found for Paper {paper_index}: {eval_file}")

            summary_lines.append(f"\n## Paper {paper_index}\n")
            summary_lines.append(f"\n{analysis}\n")
            summary_lines.append("\n---\n")

        summary_content = "\n".join(summary_lines)
        summary_path = round_dir / "error_analysis_summary.md"
        summary_path.write_text(summary_content, encoding="utf-8")

        print(f"  Summary saved to: {summary_path}")


    def _calculate_diff_char_count(self, old_text: str, new_text: str) -> int:

        import difflib

        diff = difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            lineterm=""
        )

        added_chars = 0
        deleted_chars = 0

        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                added_chars += len(line) - 1
            elif line.startswith('-') and not line.startswith('---'):
                deleted_chars += len(line) - 1

        total_changes = added_chars + deleted_chars
        return total_changes

    def _update_prompt_from_batch_errors(
        self,
        round_dir: Path,
        aggregate_metrics: Dict[str, Any]
    ) -> bool:

        print(f"\n{'='*80}")
        print("Updating Prompt Based on Batch Error Analysis")
        print(f"{'='*80}\n")

        is_first_update = (self.prompt_version == 0)
        char_limit = 700 if is_first_update else 200
        print(f"  Character limit: {char_limit} ({'first update' if is_first_update else 'subsequent update'})")

        summary_path = round_dir / "error_analysis_summary.md"
        if not summary_path.exists():
            print(f"  Error: Summary file not found: {summary_path}")
            return False

        error_summary = summary_path.read_text(encoding="utf-8")
        print(f"  Loaded error analysis summary: {summary_path}")

        current_prompt_file = self.experiment_dir / f"{self.prompt_version:02d}_prompt.md"
        if not current_prompt_file.exists():
            print(f"  Error: Current prompt file not found: {current_prompt_file}")
            return False

        current_prompt = current_prompt_file.read_text(encoding="utf-8")
        current_word_count = len(current_prompt.split())
        print(f"  Current prompt version: {self.prompt_version:02d} ({current_word_count} words)")

        update_prompt_template = """You are an expert Prompt Engineer tasked with optimizing a prompt for an LLM to extract molecular glue degradation assay measurement data from scientific papers.
Your goal is to fine-tune the existing prompt to fix extraction errors while keeping changes minimal based on the provided error analysis.

### Context
The summary of error reports for rejected {num_papers} papers is as follows:
    {error_summary}

The current prompt is as follows ({current_word_count} words):
    {current_prompt}

    
### Instructions
Analyze the general error patterns from the summary. Update the current prompt to prevent these errors using simple, clear language. Let's think step by step.

# Constraints
1. **Target Sections**: You are ONLY allowed to modify `# Goal` and `# Extraction Principles`.
2. **Frozen Sections**: Do **NOT** modify `# Required Fields` or `# Output Format` under any circumstances.
3. **Generalization**: Avoid overfitting. Identify the root cause of the errors and fix the prompt generally. Do not hardcode details about specific compounds, targets, recruiters, or cell lines found in the error report.
4. **Strict Conciseness**: You may only update or add **maximum 2 sentences** in total. Prioritize the most critical fix. The total added/modified text should be approximately ≤ {char_limit} characters.


# Output Format
Return **only** a valid, raw JSON object (no markdown code blocks, no pre-text):
{{
    "updated_prompt": "The complete updated prompt string"
}}
Your response:
"""

        prompt_kwargs = {
            "error_summary": error_summary,
            "num_papers": aggregate_metrics["num_papers"],
            "avg_p1_prec": aggregate_metrics["avg_part1_precision"],
            "avg_p1_rec": aggregate_metrics["avg_part1_recall"],
            "avg_p2_prec": aggregate_metrics["avg_part2_precision"],
            "avg_p2_rec": aggregate_metrics["avg_part2_recall"],
            "prec_threshold": aggregate_metrics["precision_threshold"],
            "rec_threshold": aggregate_metrics["recall_threshold"],
            "current_prompt": current_prompt,
            "current_word_count": current_word_count,
            "char_limit": char_limit,
        }

        final_prompt = update_prompt_template.format(**prompt_kwargs)

        print("\n" + "="*80)
        print("PROMPT UPDATE - LLM INPUT")
        print("="*80)
        print(final_prompt[:2000] + "..." if len(final_prompt) > 2000 else final_prompt)
        print("="*80 + "\n")

        from tpd_curator.llm.run_llm_api import run_llm_api

        try:
            response_text, model_name = run_llm_api(
                custom_prompt=final_prompt,
                paper_text=None,
                temperature=0.7
            )

            import re
            content = response_text.strip()
            content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.MULTILINE).strip()

            result = json.loads(content)
            updated_prompt = result.get("updated_prompt", current_prompt)

            print(f"  LLM Update Complete")

        except Exception as e:
            print(f"  ✗ Error calling LLM for prompt update: {e}")
            import traceback
            traceback.print_exc()
            return False

        char_changes = self._calculate_diff_char_count(current_prompt, updated_prompt)
        print(f"\n  Character changes: {char_changes}")

        if char_changes > char_limit:
            print(f"  ⚠ Warning: Changes exceed {char_limit} character limit ({char_changes} chars)")
            print(f"  Proceeding with update despite exceeding limit")

        if char_changes == 0:
            print(f"  No changes made to prompt - skipping update")
            return False

        self.prompt_version += 1
        new_prompt_file = self.experiment_dir / f"{self.prompt_version:02d}_prompt.md"
        new_prompt_file.write_text(updated_prompt, encoding="utf-8")
        print(f"\n  ✓ Updated prompt saved: {new_prompt_file}")

        self.current_prompt = updated_prompt

        print(f"\n  Prompt version updated: {self.prompt_version-1:02d} → {self.prompt_version:02d}")
        print(f"  Modified {char_changes} characters")
        print(f"{'='*80}\n")

        return True
