#!/usr/bin/env python3


import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tpd_curator.capo import Capo, PipelinePausedException


class TeeLogger:

    def __init__(self, file_path, stream):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Batch Active Prompting Pipeline - Process multiple papers with aggregate evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (batch_size=6, thresholds=0.9)
  python scripts/prompt_optimization/run_capo.py \\
    --config_path tpd_curator/config/config_api_local_0908.yaml \\
    --input_prompt prompts/simple_initial_prompt.md

  # Custom batch size
  python scripts/prompt_optimization/run_capo.py \\
    --config_path tpd_curator/config/config_api_local_0908.yaml \\
    --input_prompt prompts/simple_initial_prompt.md \\
    --batch_size 10

  # Custom thresholds (lower than default 0.9)
  python scripts/prompt_optimization/run_capo.py \\
    --config_path tpd_curator/config/config_api_local_0908.yaml \\
    --input_prompt prompts/simple_initial_prompt.md \\
    --precision_threshold 0.85 \\
    --recall_threshold 0.85

  # Full customization
  python scripts/prompt_optimization/run_capo.py \\
    --config_path tpd_curator/config/config_api_local_0908.yaml \\
    --input_prompt prompts/simple_initial_prompt.md \\
    --batch_size 8 \\
    --precision_threshold 0.92 \\
    --recall_threshold 0.92

  # Pause after each paper for manual review
  python scripts/prompt_optimization/run_capo.py \\
    --config_path tpd_curator/config/config_api_local_0908.yaml \\
    --input_prompt prompts/simple_initial_prompt.md \\
    --pause-after-extraction

  # Resume after fixing semantic cache (normal resume - continues to next paper)
  python scripts/prompt_optimization/run_capo.py \\
    --config_path tpd_curator/config/config_api_local_0908.yaml \\
    --input_prompt prompts/simple_initial_prompt.md \\
    --pause-after-extraction \\
    --resume

  # Verify semantic cache corrections (re-evaluate and pause again)
  python scripts/prompt_optimization/run_capo.py \\
    --config_path tpd_curator/config/config_api_local_0908.yaml \\
    --input_prompt prompts/simple_initial_prompt.md \\
    --pause-after-extraction \\
    --resume \\
    --verify-only
        """
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to pipeline configuration YAML file"
    )

    parser.add_argument(
        "--input_prompt",
        type=str,
        required=True,
        help="Path to initial extraction prompt (Markdown file)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of papers to process in batch (default: from config, typically 6)"
    )

    parser.add_argument(
        "--precision_threshold",
        type=float,
        default=None,
        help="Precision threshold for acceptance (default: from config, typically 0.9)"
    )

    parser.add_argument(
        "--recall_threshold",
        type=float,
        default=None,
        help="Recall threshold for acceptance (default: from config, typically 0.9)"
    )

    parser.add_argument(
        "--use_api",
        action="store_true",
        default=True,
        help="Use OpenRouter API (kept for compatibility; API is the only supported backend)."
    )

    parser.add_argument(
        "--with_history",
        action="store_true",
        default=True,
        help="Use 3-rounds conversation mode (default: True)"
    )

    parser.add_argument(
        "--pause-after-extraction",
        action="store_true",
        help="Pause after batch evaluation (can resume later with --resume)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a paused experiment"
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Experiment directory to resume from (optional, auto-detects most recent if not specified)"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="When used with --resume, only re-evaluate and pause again (don't continue to next paper). "
             "Useful for verifying semantic cache corrections multiple times."
    )

    args = parser.parse_args()

    if not Path(args.config_path).exists():
        print(f"Error: Config file not found: {args.config_path}")
        sys.exit(1)

    if not Path(args.input_prompt).exists():
        print(f"Error: Prompt file not found: {args.input_prompt}")
        sys.exit(1)

    print("\n" + "="*80)
    print("Batch Active Prompting Pipeline")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Config:                  {args.config_path}")
    print(f"  Initial Prompt:          {args.input_prompt}")
    print(f"  Batch Size:              {args.batch_size if args.batch_size else 'from config'}")
    print(f"  Precision Threshold:     {args.precision_threshold if args.precision_threshold else 'from config'}")
    print(f"  Recall Threshold:        {args.recall_threshold if args.recall_threshold else 'from config'}")
    print(f"  With History:            {args.with_history}")
    print(f"  Pause After Extraction:  {args.pause_after_extraction}")
    print(f"  Resume:                  {args.resume}")
    if args.experiment_dir:
        print(f"  Experiment Directory:    {args.experiment_dir}")
    if args.verify_only:
        print(f"  Verify Only:             {args.verify_only}")
    print("="*80 + "\n")

    try:
        pipeline = Capo(
            config_path=args.config_path,
            initial_prompt_path=args.input_prompt,
            batch_size=args.batch_size,
            precision_threshold=args.precision_threshold,
            recall_threshold=args.recall_threshold,
            with_history=args.with_history,
            pause_after_extraction=args.pause_after_extraction,
            resume=args.resume,
            experiment_dir_to_resume=args.experiment_dir,
            verify_only=args.verify_only,
        )

        log_file = pipeline.experiment_dir / f"batch_active.log"
        stdout_logger = TeeLogger(log_file, sys.stdout)
        stderr_logger = TeeLogger(log_file, sys.stderr)

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = stdout_logger
        sys.stderr = stderr_logger

        print(f"\nLogging to: {log_file}\n")

        if args.resume:
            from tpd_curator.utils.pause_manager import PauseManager

            existing_rounds = sorted(pipeline.experiment_dir.glob("round_*"))
            if existing_rounds:
                latest_round = max([int(r.name.split("_")[1]) for r in existing_rounds])
                latest_round_dir = pipeline.experiment_dir / f"round_{latest_round}"
                pause_manager = PauseManager(results_dir=str(latest_round_dir))
                pause_state = pause_manager.load_pause_state()

                if pause_state and pause_state.phase == "prompt_updated":
                    next_round = pause_state.batch_info.get("next_round", latest_round + 1)
                    starting_round = next_round
                    pause_manager.clear_pause_state()
                    print(f"\n[RESUME] Resuming after prompt update")
                    print(f"[RESUME] Starting Round {starting_round} with updated prompt (version {pause_state.batch_info.get('prompt_version', 'unknown')})\n")
                else:
                    starting_round = latest_round
                    print(f"\n[RESUME] Found existing round {latest_round}, will check for pause state\n")
            else:
                print("\n[RESUME] No existing rounds found, starting from round 1\n")
                starting_round = 1
        else:
            starting_round = 1

        round_num = starting_round
        max_rounds = pipeline.max_rounds
        final_results = None

        while round_num <= max_rounds:
            print(f"\n{'='*80}")
            print(f"ROUND {round_num}")
            print(f"{'='*80}\n")

            try:
                results = pipeline.run_batch_round(round_num=round_num)
                final_results = results
                batch_status = results.get("batch_status", {})
                status = batch_status.get("status", "unknown")

                if status == "accept":
                    print(f"\n✓ Batch ACCEPTED after round {round_num}!")
                    break

                if status == "reject":
                    if round_num < max_rounds:
                        print(f"\n✗ Batch REJECTED. Continuing to Round {round_num + 1}...")
                        round_num += 1
                    else:
                        print(f"\n✗ Maximum rounds ({max_rounds}) reached.")
                        break

            except PipelinePausedException as e:
                print(f"\n[PAUSED] Pipeline paused: {e}")

                final_results = {
                    "round_num": e.round_num,
                    "round_dir": e.round_dir,
                    "aggregate_metrics": e.aggregate_metrics,
                    "batch_status": {
                        "status": "paused",
                        "failing_metrics": []
                    }
                }
                break

        if final_results:
            print("\n" + "="*80)
            print("BATCH PROCESSING COMPLETE")
            print("="*80)
            print(f"Final Round:  {final_results['round_num']}")
            print(f"Status:       {final_results['batch_status']['status'].upper()}")
            print(f"Results Dir:  {final_results['round_dir']}")
            print(f"\nAggregate Metrics:")
            print(f"  Part 1 Precision: {final_results['aggregate_metrics']['avg_part1_precision']:.4f}")
            print(f"  Part 1 Recall:    {final_results['aggregate_metrics']['avg_part1_recall']:.4f}")
            print(f"  Part 2 Precision: {final_results['aggregate_metrics']['avg_part2_precision']:.4f}")
            print(f"  Part 2 Recall:    {final_results['aggregate_metrics']['avg_part2_recall']:.4f}")

            if final_results['batch_status']['failing_metrics']:
                print(f"\nFailing Metrics:")
                for metric in final_results['batch_status']['failing_metrics']:
                    value = final_results['aggregate_metrics'][metric]
                    threshold = final_results['aggregate_metrics']['precision_threshold'] if 'precision' in metric else final_results['aggregate_metrics']['recall_threshold']
                    print(f"  {metric}: {value:.4f} < {threshold}")

            print(f"\nProcessed Papers: {final_results['aggregate_metrics']['num_papers']}")
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")

        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_logger.close()
        stderr_logger.close()

        if final_results:
            print(f"\n✓ Batch processing complete!")
            print(f"  Status: {final_results['batch_status']['status'].upper()}")
            print(f"  Results: {final_results['round_dir']}")
            print(f"  Log: {log_file}\n")

            sys.exit(0 if final_results['batch_status']['status'] == 'accept' else 1)
        else:
            print(f"\n⚠ Batch processing stopped (paused or interrupted)")
            print(f"  Log: {log_file}\n")
            sys.exit(2)

    except KeyboardInterrupt:
        print("\n\nBatch processing interrupted by user.")
        sys.exit(130)

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
