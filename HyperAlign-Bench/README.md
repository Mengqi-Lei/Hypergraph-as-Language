# HyperAlign-Bench

This directory contains the benchmark construction and data utility code for **HyperAlign-Bench**.

Formal release tasks:

- **VC**: vertex classification
- **HEC**: hyperedge classification

## Data Bundle

The full HyperAlign-Bench data bundle download links are listed in `../DOWNLOADS.md`.

After extraction, each dataset directory should contain:

```text
processed_data.pt
meta.json
samples/
embeddings/qwen3emb_0.6b/
overview/qwen3emb_0.6b/
```

Expected local layout:

```text
HyperAlign-Bench/dataset/
├── arxiv_hg/
├── cora_cc/
├── pubmed/
├── dblp/
└── imdb/
```

## Dataset Summary

| Dataset | Directory | Vertices | Hyperedges | Classes | Use |
|---|---|---:|---:|---:|---|
| Arxiv-HG | `arxiv_hg` | 169,343 | 123,826 | 40 | train + in-domain test |
| Cora-CC | `cora_cc` | 2,341 | 2,219 | 7 | zero-shot |
| PubMed | `pubmed` | 19,716 | 13,011 | 3 | zero-shot |
| DBLP | `dblp` | 2,591 | 2,463 | 6 | zero-shot |
| IMDB | `imdb` | 3,939 | 839 | 3 | zero-shot |

## Build Notes

When building Arxiv-HG from raw OGBN-Arxiv, the builder imports HIDT metadata from the runtime. Either place `Hyper-Align` next to this directory or set:

```bash
export HYPERALIGN_ROOT=/path/to/Hyper-Align
```

The Arxiv-HG construction uses source-excluded co-citation hyperedges: for each source paper, the set of cited papers forms one hyperedge, and the source paper itself is excluded from the member set.

## Public Scripts

- `scripts/build_arxiv_hypergraph_dataset.sh`: build Arxiv-HG metadata and raw VC/HEC task samples from OGBN-Arxiv source files.
- `scripts/build_dataset_text_embeddings.sh`: build node/hyperedge text embeddings for a prepared dataset package.
