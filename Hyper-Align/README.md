# Hyper-Align

This directory contains the model/runtime code for **Hyper-Align**.

## What Is Included

- `model/`: LLM wrappers and the HIP projector
- `utils/`: HIDT-O serialization, overview features, prompt/data utilities
- `train/`: supervised projector-tuning pipeline
- `eval/`: generation/evaluation entrypoints
- `scripts/`: data preparation, training, and evaluation wrappers
- `tests/`: lightweight HIDT/projector checks

Large assets are not included. See `../DOWNLOADS.md`.

## Main Scripts

```bash
# Prebake HyperAlign-Bench VC samples
python scripts/prebake_data.py \
  --hyper-data-root ../HyperAlign-Bench/dataset/arxiv_hg \
  --task nc \
  --split test \
  --center-kind vertex \
  --pretrained-embedding-type qwen3emb_0.6b \
  --enable-text-context

# Joint train on Arxiv-HG VC + HEC
MODEL_BASE=Qwen/Qwen3-8B \
bash scripts/train_arxiv_hg_joint_vc_hec_2ep.sh

# Evaluate one split with the main checkpoint
BASE_MODEL=Qwen/Qwen3-8B \
bash scripts/evaluate_arxiv_hg_vc.sh \
  ./checkpoints/hyper-align-qwen3-8b-qwen3emb0.6b-hidt-o-hip-joint2ep \
  0 100 smoke
```

Public evaluation wrappers follow `scripts/evaluate_<dataset>_<task>.sh`, where
`dataset` is one of `arxiv_hg`, `cora_cc`, `pubmed`, `dblp`, `imdb`, and `task`
is `vc` or `hec`.

## Important Defaults

- `EVAL_DTYPE=bf16`
- `EVAL_BATCH_SIZE=1`
- `PRETRAINED_EMBEDDING_TYPE=qwen3emb_0.6b`
- `MAX_HYPERGRAPH_TOKENS=160`
- `MAX_INCIDENT_HYPEREDGES=8`
- `MAX_MEMBERS_PER_HYPEREDGE=8`
- `OVERVIEW_HOPS=2`
- `OVERVIEW_ORDER_BUCKETS=4`
- `LAMBDA_ORD=0.01`
- `LAMBDA_REL=0.01`

Set `LAMBDA_ORD=0.0` and `LAMBDA_REL=0.0` only when explicitly disabling
the auxiliary consistency losses.
