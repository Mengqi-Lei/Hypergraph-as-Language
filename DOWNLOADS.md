# Downloads

Large assets are kept out of git. The Hyper-Align checkpoint is distributed through Hugging Face, and HyperAlign-Bench is distributed as a Hugging Face dataset. Base LLMs and embedding models are not mirrored by us; use their official sources.

## Hyper-Align Checkpoints

| Asset | Description | Link |
|---|---|---|
| Hyper-Align-Qwen3-8B-qwen3emb0.6b-HIDT-O-HIP-joint2ep | Main Hyper-Align projector/checkpoint trained jointly on Arxiv-HG VC and HEC for 2 epochs. | [MengqiLei/hyper-align](https://huggingface.co/MengqiLei/hyper-align) |

Recommended checkpoint location after download:

```text
Hyper-Align/checkpoints/hyper-align-qwen3-8b-qwen3emb0.6b-hidt-o-hip-joint2ep/
```

## External Base Models And Encoders

These models are required only through their official sources. Please follow the licenses and access requirements of the corresponding model providers.

| Model / Resource | Official Link |
|---|---|
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Qwen3-Embedding-0.6B | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Qwen3-Embedding-4B | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Vicuna-7B-v1.5 | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| LLaMA-family models | [meta-llama](https://huggingface.co/meta-llama) |
| SBERT all-MiniLM-L6-v2 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

## HyperAlign-Bench Datasets

Download the full HyperAlign-Bench data bundle from Hugging Face:

| Asset | Contents | Link |
|---|---|---|
| HyperAlign-Bench full data bundle | Arxiv-HG, Cora-CC, PubMed, DBLP, and IMDB with `processed_data.pt`, `meta.json`, task samples, `qwen3emb_0.6b` embeddings, and overview features. | [MengqiLei/hyperalign-bench](https://huggingface.co/datasets/MengqiLei/hyperalign-bench) |

After extraction, each dataset directory should contain:

```text
processed_data.pt
meta.json
samples/
embeddings/qwen3emb_0.6b/
overview/qwen3emb_0.6b/
```

Recommended dataset location after download:

```text
HyperAlign-Bench/dataset/<dataset-name>/
```
