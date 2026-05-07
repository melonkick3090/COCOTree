# COCOTree Pipeline

This repository contains the code used to run the teacher pipeline for COCOTree-style hierarchical mask decomposition. It takes input images, proposes object and part prompts with a language model, runs SAM3 masks for those prompts, and writes a tree-structured output for each image.

Dataset: https://huggingface.co/datasets/melonkick/COCOTree

## Setup

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

The pipeline expects the SAM3 Python package to be importable. Place the SAM3 checkpoint at `models/sam3/sam3.pt`, or pass `--checkpoint_path`. You can also set `SAM3_MODELS_DIR` to a directory containing `sam3/sam3.pt`.

Set an OpenRouter key before running:

```bash
export OPENROUTER_API_KEY="..."
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="..."
```

No API keys are stored in this repository.

## Run

```bash
python -m cocotree_pipeline.run_pipeline \
  --image_dir path/to/images \
  --out_dir runs/cocotree \
  --run_name demo
```

Useful options:

```bash
--recursive
--start_idx 0 --end_idx 99
--shard_id 0 --num_shards 4
--skip_existing --no-overwrite
--checkpoint_path path/to/sam3.pt
--model_name google/gemini-3-flash-preview
```

Each image gets its own output directory. The main outputs include the tree JSON, mask files, candidate/debug artifacts when enabled, provenance metadata, and a `_DONE.json` marker for completed images.

## Code Layout

```text
cocotree_pipeline/
  run_pipeline.py       command-line entry point
  generator.py          hierarchical decomposition pipeline
  sam3_backend.py       SAM3 wrapper
  llm_openrouter.py     OpenRouter client and response parsing
  prompts.py            prompt templates
  mask_utils.py         mask geometry utilities
  mask_postprocess.py   mask cleanup and subcrop splitting
  image_utils.py        image IO and visualization helpers
  io_utils.py           output serialization helpers
  types.py              configuration and result dataclasses
```

## Data

The released annotations are hosted on Hugging Face:

https://huggingface.co/datasets/melonkick/COCOTree

This code release is intended to make the data construction pipeline inspectable and runnable for review and follow-up experiments.
