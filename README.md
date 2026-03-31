# TraceGen

**TraceGen** is a framework for synthesizing realistic, validated software bugs from seed defects. It extracts *DefectChains* from real bugs, discovers structurally analogous locations via graph-based matching, and generates new bugs with calibrated-difficulty problem statements.

## Overview

TraceGen operates in three stages:

1. **DefectChain Extraction (Stage 1)** — Extracts a localization chain (symptom → root cause path over the code graph) and a repair chain (fix intents from AST diff) from each seed bug.
2. **Agentic Synthesis (Stage 2)** — Discovers candidate injection points via vector similarity + depth matching + intent compatibility, then uses a ReAct-style LLM agent to synthesize a new bug at the selected location. Each synthesized bug is validated in an isolated container.
3. **Problem Statement Design (Stage 3)** — Generates three difficulty levels (L1/hard, L2/medium, L3/easy) by selectively exposing nodes of the localization chain.

Evaluation of synthesized bugs is performed externally using [SWE-agent](https://github.com/princeton-nlp/SWE-agent). See `scripts/sweagent/` for conversion and evaluation scripts.

## Synthesized Benchmark

The synthesized benchmark is located in `data/synthesized_bugs/`:

| File | Description |
|------|-------------|
| `tracegen_full.json` | Validated synthetic bugs in SWE-bench format, with injection patches, test oracles, and seed instance references |
| `problem_statements_levels.json` | Problem statements at three difficulty levels (L1/L2/L3) + original for all instances |
| `valid_summary.json` | Validation summary for all valid bugs (p2f/p2p counts) |
| `quality_metrics.json` | Per-instance quality scores across multiple dimensions |
| `chain_guided_ps.json` | Chain-guided PS generation with entropy/signal metrics |
| `sample.json` | Representative instances with full patches for quick inspection |

Showcase examples with end-to-end detail are in `data/examples/showcase.json` (selected instances from different repositories, each with injection patch + three PS levels).

### Baseline comparisons: `data/baselines/`

Evaluation results from `data/evaluation/` include SWE-agent resolve rates across models and PS difficulty levels.

## Installation

```bash
# Python 3.10+
pip install -e .

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
# Run with default config (sample instances)
python main.py

# Run with a specific config
python main.py --config-name config_full_run

# Hydra overrides
python main.py runtime.batch_size=20 method.ablation.disable_graph_matching=true

# Validation-only mode (skip Stage 1-2, re-validate existing outputs)
python main.py runtime.validation_only_dir=outputs/<run-dir>
```

### SWE-agent Evaluation

```bash
# Convert synthesized bugs to SWE-agent format
python scripts/sweagent/convert_to_sweagent.py --input data/synthesized_bugs/tracegen_full.json --output sweagent_instances/

# Generate PS at different difficulty levels
python scripts/sweagent/generate_ps_levels.py --input sweagent_instances/instances_original.json

# Evaluate with SWE-agent
python scripts/sweagent/evaluate_results.py --config scripts/sweagent/tracegen_config.yaml
```

## Configuration

Main config: `configs/config.yaml`. Example configs for ablation studies and full runs are in `configs/examples/`.

Key settings:
- `synthesis_llm` / `analyzer_llm` — Any OpenAI-compatible endpoint
- `method.ablation.*` — Ablation switches for component analysis
- `method.synthesis.agent.ps_select` — PS difficulty level (L1/L2/L3)
- `validation.*` — Container-based validation settings

## Project Structure

```
├── main.py                     # Hydra entry point
├── configs/                    # Hydra configurations
│   ├── config.yaml             # Default config
│   └── examples/               # Ablation and full-run configs
├── data/
│   ├── seed_instances/         # Input SWE-bench seed bugs
│   ├── synthesized_bugs/       # Synthesized benchmark (validated bugs)
│   ├── examples/               # Showcase examples with full detail
│   ├── baselines/              # Baseline comparison results
│   └── evaluation/             # SWE-agent evaluation results
├── src/
│   ├── core/                   # Data structures, interfaces, repo profiles
│   ├── graph/                  # AST-based code graph construction
│   ├── modules/
│   │   ├── extraction/         # Stage 1: DefectChain extraction
│   │   ├── synthesis/          # Stage 2: Agentic bug synthesis
│   │   ├── validation/         # Stage 3: Container-based validation
│   │   └── localization/       # Fault localization utilities
│   └── pipeline/               # Pipeline orchestration
├── scripts/
│   ├── quality/                # Quality metric computation and figures
│   └── sweagent/               # SWE-agent conversion and evaluation
└── tools/                      # Embedding generation and data conversion
```

## Requirements

- Python >= 3.10
- Docker or Podman (for Stage 3 validation)
- OpenAI-compatible LLM API access

## License

MIT License
