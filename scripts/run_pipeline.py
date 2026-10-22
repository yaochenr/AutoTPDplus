import argparse
import sys
import yaml
from pathlib import Path
from multiprocessing import freeze_support
from datetime import datetime

from tpd_curator.pipeline import *


class TeeLogger:

    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def setup_logging(results_dir: Path):

    log_file_path = results_dir / "pipeline_run.log"

    log_file = open(log_file_path, 'w', encoding='utf-8', buffering=1)

    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)

    print(f"{'='*80}")
    print(f"Pipeline Run Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file_path}")
    print(f"{'='*80}\n")

    return log_file


def main(args):
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    config_file = load_config(args.config_path)
    results_base_dir = Path(config_file["results"]["results_dir"])
    results_dir = results_base_dir / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = setup_logging(results_dir)

    try:
        config_yaml_path = results_dir / "config.yaml"
        with open(config_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_file, f, allow_unicode=True, sort_keys=False)
        print(f"Saved config to {config_yaml_path}\n")

        config_file["results"]["current_run_dir"] = str(results_dir)
        config_file["results"]["timestamp"] = timestamp

        if args.cleaning:
            try:
                print("Processing data cleaning...")

                if "json_path" in config_file["data"]:
                    print("Processing JSON data cleaning...")
                    clean_json_papers_batch(config_file)
                else:
                    print("No json_path specified in config. Skipping JSON cleaning.")

            except Exception as e:
                print(f"Data cleaning failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            print("Skip data cleaning step.")



        input_prompt_content = None
        if args.input_prompt:
            prompt_path = Path(args.input_prompt)
            if not prompt_path.exists():
                print(f"Error: The specified markdown file does not exist - {args.input_prompt}")
                return 1
            with open(prompt_path, "r", encoding="utf-8") as f:
                input_prompt_content = f.read()
        else:
            print("Skip input custom LLM prompt from markdown file.")

        if args.llm:
            try:
                if args.with_history:
                    print("Using LLM API with memory history.")
                    run_llm_pipeline(config_file, read_papers=True, md_prompt=input_prompt_content, with_history=True)
                else:
                    print("Using LLM API without memory history.")
                    run_llm_pipeline(config_file, read_papers=True, md_prompt=input_prompt_content)
            except Exception as e:
                print(f"Reasoning failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            print("Skip LLM reasoning step.")

        csv_file_path = None
        if args.csv:
            try:
                csv_file_path = save_csv(config_file=config_file, results_dir=args.results_dir)
            except Exception as e:
                print(f"Save CSV failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            print("Skip saving LLM output to CSV file.")

        if csv_file_path:
            try:
                processed_csv_path = process_units(csv_file_path)
                print(f"Finish processing the units and save the results in: {processed_csv_path}")
            except Exception as e:
                print(f"Fail to process units: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Skip processing concentration units.")

        if args.smiles:
            print("\nFetching SMILES from multiple sources...")
            if locals().get('processed_csv_path'):
                try:
                    use_chembl = not args.no_chembl
                    sources = "PubChem + ChEMBL" if use_chembl else "PubChem only"
                    print(f"Data sources: {sources}")

                    smiles_csv_path = fetch_smiles_multi_source(
                        csv_file_path=processed_csv_path,
                        use_chembl=use_chembl,
                        rate_limit_pubchem=0.2,
                        rate_limit_chembl=0.5
                    )
                    print(f"✓ Added SMILES to: {smiles_csv_path}")
                except Exception as e:
                    print(f"✗ SMILES fetch failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Skip SMILES fetch: no processed CSV file available")
                print("Please run with --csv first to generate the processed CSV")

        if args.accuracy:
            print("Calculating extraction accuracy...")
            try:
                if args.pred:
                    pred_csv_path = args.pred
                else:
                    pred_csv_path = processed_csv_path if locals().get('processed_csv_path') else None

                labeled_csv_path = args.labeled if args.labeled else None

                eval_results = run_evaluation(
                    config_file=config_file,
                    pred_csv_path=pred_csv_path,
                    labeled_csv_path=labeled_csv_path,
                    lenient_mode=args.lenient_mode,
                    output_dir=args.eval_output_dir,
                    ternary_complex_level=args.ternary_complex_level
                )

                if not eval_results:
                    print("Evaluation could not be completed; please check the config and data files.")
            except Exception as e:
                print(f"Error while computing accuracy: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Skip calculating extraction accuracy.")

        if args.llm_vs_baseline:
            print("Running LLM vs Baseline evaluation...")
            try:
                if args.pred:
                    llm_csv_path = args.pred
                else:
                    llm_csv_path = processed_csv_path if locals().get('processed_csv_path') else None

                baseline_csv_path = args.baseline if args.baseline else None

                eval_results = run_llm_vs_baseline_evaluation(
                    config_file=config_file,
                    llm_csv_path=llm_csv_path,
                    baseline_csv_path=baseline_csv_path,
                    output_dir=args.eval_output_dir,
                )

                if not eval_results:
                    print("LLM vs Baseline evaluation failed. Please check configuration and data files.")
            except Exception as e:
                print(f"LLM vs Baseline evaluation error: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Skip LLM vs Baseline evaluation.")

        print(f"\n{'='*80}")
        print(f"PIPELINE EXECUTION COMPLETED")
        print(f"{'='*80}")
        print(f"Results saved in: {results_dir}")
        print(f"Log file: {results_dir / 'pipeline_run.log'}")
        print(f"Execution finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"PIPELINE EXECUTION FAILED")
        print(f"{'='*80}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
    finally:
        if 'log_file' in locals():
            log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Run molecular glue data extraction pipeline.")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file.")
    parser.add_argument("--convert", action="store_true", help="Whether to execute PDF conversion.")
    parser.add_argument("--cleaning", action="store_true", help="Clean the converted data and save it to JSONL file.")
    parser.add_argument("--llm", action="store_true", help="Using LLM to reason.")
    parser.add_argument("--with_history", action="store_true", help="Use LLM API with memory history.")
    parser.add_argument("--csv", action="store_true", help="Save all LLM output to a CSV file.")
    parser.add_argument("--accuracy", action="store_true", help="Calculate the accuracy of the extraction results.")
    parser.add_argument("--llm_vs_baseline", action="store_true", help="Compare LLM extraction results against a baseline CSV using the same evaluation logic as --accuracy.")
    parser.add_argument("--baseline", type=str, help="Baseline CSV file path for --llm_vs_baseline (optional, will use evaluation.baseline_data_path in config by default)")
    parser.add_argument("--lenient_mode", action="store_true", help="Use lenient evaluation mode (allows Assay/Cell_Line to be empty in predictions)")
    parser.add_argument("--ternary_complex_level", action="store_true",
                       help="Evaluate at ternary complex level: record key = {Compound_Name, Degradation_Target, Recruiter}; field evaluation adds Assay and Cell_Line.")
    parser.add_argument("--use_api", action="store_true", help="Use OpenRouter API (kept for compatibility; API is the only supported backend).")
    parser.add_argument("--pred", type=str, help="Prediction CSV file path (optional, will use the latest processed CSV by default)")
    parser.add_argument("--labeled", type=str, help="Labeled CSV file path (optional, will use the path specified in config by default)")
    parser.add_argument("--input_prompt", type=str, help="Input LLM prompt from a Markdown file. In Active Prompting mode, this serves as the initial prompt for optimization.")
    parser.add_argument("--pmc", action="store_true", help="Process PMC data instead of PDFs.")
    parser.add_argument("--data_type", type=str, choices=["pdf", "pmc", "auto"], default="auto",
                       help="Specify data type to process (pdf, pmc, or auto-detect).")
    parser.add_argument("--results_dir", type=str, help="Specify results directory to process (e.g., /path/to/results/251024_1640). If not specified, uses the latest results directory.")
    parser.add_argument("--eval_output_dir", type=str, help="Specify output directory for evaluation result files (JSON, detailed results, need_review). Defaults to the same directory as --pred.")
    parser.add_argument("--smiles", action="store_true", help="Fetch SMILES representations from PubChem and ChEMBL based on IUPAC names or compound names.")
    parser.add_argument("--no-chembl", action="store_true", help="Skip ChEMBL queries (only use PubChem for SMILES lookup).")

    args = parser.parse_args()
    
    if not Path(args.config_path).exists():
        print(f"Error: The specified configuration file does not exist - {args.config_path}")
        exit(1)

    sys.exit(main(args) or 0)