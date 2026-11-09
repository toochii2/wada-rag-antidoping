# Anti-doping medication queries: benchmark and replication code

Replication package for a study that evaluates how local large language models answer athletes' questions about whether a medicine is allowed in sport. Models answer with and without retrieval over the World Anti-Doping Agency (WADA) Prohibited List, and with retrieval over current and outdated versions of the List.

Authors: Hawraa Alsayegh (Kuwait Ministry of Health) and Ali Alsayegh (University of Strathclyde). The accompanying paper is under review; a link will be added on publication.

## Contents

| Path | Contents |
|---|---|
| `benchmark/questions.csv` | All 264 questions, one row each: the question as an athlete would ask it, the gold verdict, any also-accepted verdict, required conditions and the List clause |
| `benchmark/items.jsonl`, `benchmark/items_status_change.jsonl` | The same questions with all fields (236 main items in groups A to D; 28 items on two historical status changes, group H) |
| `benchmark/corrections.csv` | Every change made to questions, gold labels, accepted answers and one scoring rule after the first runs, with the value before and after and the reason |
| `benchmark/brand_map.csv`, `benchmark/subset_consistency.json` | Brand-to-ingredient mapping used for brand-name items; item subset for the repeated-sampling run |
| `prompts/v1/` | The system and user prompts and the JSON output schema used for every call |
| `src/` | List parsing, hybrid retrieval (BM25 + bge-m3, reciprocal rank fusion), model calls, scoring and analysis; also the rule-table and question generators |
| `scripts/` | Run scripts for all model calls |
| `results/runs/` | Model outputs for every call: verdict, cited clause references and retrieval record |
| `results/run_manifest.json` | Model digests, parameters, seeds and dates for every run |

Outputs for the original wording of the twelve reworded questions remain in `results/runs/`, marked by their question text; scoring uses only the current wording.

Gold verdicts are one of `prohibited_at_all_times`, `prohibited_in_competition_only`, `prohibited_in_particular_sports`, `permitted`, `permitted_with_conditions` and `cannot_determine_refer`. They are derived from the List text by documented rules.

## Not included

- **WADA documents.** The Prohibited Lists are © World Anti-Doping Agency and are not redistributed. Download them from https://www.wada-ama.org/en/resources/world-anti-doping-code-and-international-standards/prohibited-list.
- **Text quoted from the List.** The released outputs keep each answer's verdict and cited clause references. The models' free-text explanations are removed, because they often quote the List. The facts that scoring reads from that text (condition completeness, mention of a therapeutic use exemption, out-of-competition-only wording, citation match) are kept as flags in each output (`parsed.text_flags`), so scoring the released outputs reproduces every reported measure.

## Replicating the study

Requirements: Python 3.11 or later; [Ollama](https://ollama.com) with `qwen3:30b-a3b`, `qwen3:14b`, `gemma3:12b`, `llama3.1:8b` and `bge-m3`. The original runs used 4-bit quantised models on an Apple M4 Pro with 24 GB of memory.

1. Put the WADA Prohibited List PDFs in `data/raw/` as `2026_list_clean.pdf` and `2027_list_clean.pdf`, and in `data/raw/historical/` as `2021_list_clean.pdf` and `2023_list_clean.pdf`.
2. Set up the environment:
   ```bash
   make setup
   ```
3. Build the retrieval index, run or reuse the model calls, then score and analyse:
   ```bash
   make all
   ```

`make all` reuses the released outputs in `results/runs/` and calls a model only for items that are missing. To re-run every call, delete `results/runs/` first. Local quantised models are not bit-for-bit deterministic across hardware, so a full re-run may differ slightly. `make benchmark` regenerates the questions from the Lists. This step also needs the 2027 Monitoring Program PDF in `data/raw/`, as `2027_monitoring_program.pdf`.

## Citation and licence

Please cite this repository using [`CITATION.cff`](CITATION.cff).

- Code: MIT ([`LICENSE`](LICENSE)).
- Questions, labels and model outputs: CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).
