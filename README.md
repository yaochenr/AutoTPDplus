<h1 align="center">
  Agentic Literature Extraction Augments Targeted Protein Degradation Databases
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.11221"><img src="https://img.shields.io/badge/Paper-arXiv%3A2605.11221-b31b1b?logo=arxiv" alt="Paper"></a>
  <a href="https://yaochenr.github.io/LLM-TPD-Extraction/"><img src="https://img.shields.io/badge/Project-Website-2a78d6" alt="Project website"></a>
  <a href="https://huggingface.co/datasets/ailab-bio/AutoTPDplus"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-ffd21e" alt="Hugging Face dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-555555" alt="MIT code license"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/Data-CC%20BY%204.0-6b8e23" alt="CC BY 4.0 data license"></a>
</p>

This repository implements an end-to-end agentic workflow for extracting structured targeted protein degradation (TPD) assay records from scientific literature. The workflow integrates literature retrieval, cross-validated prompt optimization, multi-pass LLM extraction, domain-specific standardization, evaluation against expert annotations, and continuous database maintenance.

The implementation accompanies [*Beyond Manual Curation: Augmenting Targeted Protein Degradation Databases via Agentic Literature Extraction Workflows*](https://arxiv.org/abs/2605.11221) and produces two data resources: **AutoTPD**, the frozen dataset used for method evaluation, and **AutoTPD+**, a living database designed for continued literature-based updates. Both resources are available in this GitHub repository, through the [project website](https://yaochenr.github.io/LLM-TPD-Extraction/), and on [Hugging Face](https://huggingface.co/datasets/ailab-bio/AutoTPDplus).

## Pipeline

```text
Source DOIs
    → PMC full text and supplementary files
    → CAPO prompt optimization
    → frozen-prompt LLM extraction
    → post-processing and standardization
    → evaluation and data release
    → monthly literature updates
```

| Stage | Function | Main entry point |
|---|---|---|
| Literature acquisition | Retrieve and parse PMC articles, tables, and supplementary files | `tpd_curator/pmc_miner/` |
| Prompt optimization | Optimize the extraction prompt with CAPO using expert-annotated papers | `scripts/prompt_optimization/run_capo.py` |
| LLM extraction | Extract and verify assay records through multiple article passes | `scripts/run_pipeline.py` |
| Post-processing | Normalize values, remove duplicates, and enrich chemical identifiers | `scripts/post_processing.py` |
| Standardization | Standardize compounds, proteins, assays, and cell lines | `scripts/data_standardization.py` |
| Evaluation | Compare extracted records with expert ground truth and source databases | `tpd_curator/pipeline.py` |
| Monthly update | Discover new papers, extract records, merge releases, and run QC | `scripts/monthly/run_monthly_workflow.py` |

CAPO (Cross-validated Agentic Prompt Optimization) uses Extractor, Analyst, and Refiner agents to optimize the extraction instructions on a small expert-annotated cohort. The selected Champion Prompt is then fixed for full-cohort extraction, separating prompt development from final evaluation.

## Data releases

**AutoTPD** is the frozen, standardized output used for method evaluation in the accompanying study. **AutoTPD+** extends this resource into a living database of PROTAC and molecular glue degradation assays by integrating literature-extracted records with PROTAC-DB 3.0 and MolGlueDB.

The released records include standardized compound, target, recruiter, assay, cell-line, DC50, and Dmax information. **AutoTPD+ v1.0** comprises:

- **7,622** assay records
- **1,008** source publications
- **2,408** unique chemical structures
- **2 modalities:** PROTAC and molecular glue

| File | Description | Records |
|---|---|---:|
| `AutoTPDplus-PROTAC.csv` | AutoTPD+ PROTAC records | 6,766 |
| `AutoTPDplus-MG.csv` | AutoTPD+ molecular glue records | 856 |
| `AutoTPD-PROTAC.csv` | Frozen PROTAC benchmark | 714 |
| `AutoTPD-MG.csv` | Frozen molecular glue benchmark | 246 |

The data can be downloaded from [`data/release/`](data/release/), the [project website](https://yaochenr.github.io/LLM-TPD-Extraction/), or Hugging Face:

```python
from datasets import load_dataset

protac = load_dataset("ailab-bio/AutoTPDplus", "autotpdplus-protac", split="train")
molecular_glue = load_dataset("ailab-bio/AutoTPDplus", "autotpdplus-mg", split="train")
```

## Data format

Each row represents one degradation assay record.

<details>
<summary><strong>All columns</strong></summary>

| Column | Description |
|---|---|
| **IDENTIFIERS** | |
| `AutoTPD_ID` | Stable row identifier, unchanged across monthly releases |
| `Record_ID` | Record ID in the source database; empty for LLM-extracted records |
| `Compound_ID` | Compound ID in the source database; empty for LLM-extracted records |
| `DOI` | Source publication |
| **COMPOUND** | |
| `Compound_Name` | Compound name or label as reported in the paper |
| `IUPAC_Name` | IUPAC name |
| `IUPAC_Name_Source` | Where the IUPAC name came from (`PAPER`, `PUBCHEM`) |
| `SMILES` | RDKit canonical SMILES |
| `SMILES_Source` | Where the SMILES came from (`PAPER`, `PUBCHEM`, `OPSIN`, source database) |
| `StdInChIKey` | Standard InChIKey computed by RDKit from the standardized canonical SMILES |
| `Connectivity_Key` | First block of the InChIKey (connectivity only, ignores stereochemistry) |
| **DEGRADATION TARGET** | |
| `Degradation_Target` | Target as reported in the paper |
| `Degradation_Target_Gene` | HGNC gene symbol |
| `Degradation_Target_Uniprot` | UniProt accession |
| `Degradation_Target_Uniprot_MutationID` | UniProt variant ID when a specific mutant is targeted |
| `Degradation_Target_Sequence` | UniProt protein sequence |
| **RECRUITER** | |
| `Recruiter` | Recruiter (usually an E3 ligase) as reported |
| `Recruiter_Gene` | Gene symbol |
| `Recruiter_Uniprot` | UniProt accession |
| `Recruiter_Sequence` | UniProt protein sequence |
| **ASSAY CONTEXT** | |
| `Assay` | Assay as described in the paper |
| `Assay_STD` | Standardized assay family |
| `Cell_Line` | Cell line as reported, or cell-free |
| `Cell_Line_STD` | Cellosaurus recommended name |
| `Cell_Line_STD_Accession` | Cellosaurus accession (RRID) |
| `Cell_Line_Modification` | Engineered construct, e.g. a HiBiT knock-in |
| `Cell_Line_Species` | Species of the cell line |
| **DC50** | |
| `DC50` | Concentration giving 50% degradation |
| `DC50_units` | Units of DC50 |
| `DC50_h` | Treatment time in hours |
| `DC50_operator` | Qualifier: `<`, `>`, `~`, `<=`, `>=` |
| `DC50_range_min` | Range low end, or mean - error |
| `DC50_range_max` | Range high end, or mean + error |
| **DMAX** | |
| `Dmax` | Maximal degradation, in % |
| `Dmax_h` | Treatment time in hours |
| `Dmax_operator` | Qualifier: `<`, `>`, `~`, `<=`, `>=` |
| `Dmax_range_min` | Range low end, or mean - error |
| `Dmax_range_max` | Range high end, or mean + error |
| **DMAX CONCENTRATION** | |
| `Dmax_conc` | Concentration at which Dmax was measured |
| `Dmax_conc_units` | Units of Dmax_conc |
| `Dmax_conc_operator` | Qualifier for Dmax_conc |
| `Dmax_conc_range_min` | Range low end |
| `Dmax_conc_range_max` | Range high end |

</details>

## Installation

Requires Python 3.10 and [Pixi](https://pixi.sh/).

```bash
git clone https://github.com/yaochenr/LLM-TPD-Extraction.git
cd LLM-TPD-Extraction
pixi install
```

Create `.env` in the repository root and update the local paths in `tpd_curator/config/*.yaml`:

```dotenv
OPENROUTER_API_KEY=<YOUR_API_KEY>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=openai/gpt-5
```

## Running the pipeline

### Extract literature records

```bash
pixi run python scripts/run_pipeline.py \
  --config_path tpd_curator/config/llm_extraction_config_mg.yaml \
  --input_prompt prompts/champion_prompt_glues.md \
  --cleaning --llm --with_history --use_api --csv
```

For PROTAC extraction, use `llm_extraction_config_protac.yaml` and `champion_prompt_protacs.md`.

### Post-process and standardize

```bash
pixi run python scripts/post_processing.py \
  --input_csv <RUN_CSV> --output_csv <POST_CSV>

pixi run python scripts/data_standardization.py \
  --input_csv <POST_CSV> --output_csv <STANDARDIZED_CSV> \
  --record_id_prefix AUTOTPD-MG
```

### Evaluate

```bash
pixi run python scripts/run_pipeline.py \
  --config_path tpd_curator/config/evaluation_glues_config.yaml \
  --accuracy --use_api \
  --pred <PREDICTION_CSV> --labeled <GROUND_TRUTH_CSV>
```

By default, records are matched by compound, target, recruiter, assay, and cell line. Add `--ternary_complex_level` to match by compound, target, and recruiter.

### Optimize the prompt with CAPO

```bash
pixi run python scripts/prompt_optimization/run_capo.py \
  --config_path <CAPO_CONFIG_YAML> \
  --input_prompt prompts/initial_prompt.md \
  --batch_size 6 --precision_threshold 0.9 --recall_threshold 0.9
```

### Run a monthly update

```bash
pixi run python scripts/monthly/run_monthly_workflow.py run \
  --month YYYY-MM --no_pr
```

The monthly workflow runs PMC discovery, extraction, post-processing, standardization, DOI-level merging, and quality control.

## Reproducibility

Paper tables and figures can be reproduced without rerunning LLM extraction:

- `data/evaluation/`: standardized baselines, expert ground truth, LLM outputs, and manual audits;
- `semantic_cache/`: expert-reviewed semantic-matching decisions;
- `notebooks/data-processing/` and `notebooks/visualization/`: analysis and figure generation;
- `scripts/tack/`: AutoTPD+ surrogate-model benchmarks using TACK.

## Repository structure

```text
tpd_curator/       Core pipeline, CAPO, LLM client, PMC miner, and configuration
scripts/           Extraction, curation, evaluation, and monthly-update entry points
prompts/           Seed and Champion Prompts for molecular glues and PROTACs
data/              Released datasets, evaluation data, and update artifacts
semantic_cache/    Reviewed semantic-matching decisions
notebooks/         Data processing and paper analyses
docs/              Project website
```

## Citation

```bibtex
@article{rao2026beyond,
  title={Beyond Manual Curation: Augmenting Targeted Protein Degradation Databases via Agentic Literature Extraction Workflows},
  author={Rao, Yaochen and Jalalypour, Farzaneh and Krishnan, NM and Mercado, Rocío},
  journal={arXiv preprint arXiv:2605.11221},
  year={2026}
}
```

## License

Code is released under the [MIT License](LICENSE). Structured datasets are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Upstream resources retain their original licenses.
