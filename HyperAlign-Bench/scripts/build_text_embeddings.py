from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Encoder backends
# ---------------------------------------------------------------------------

class SBERTEncoder:
    """sentence-transformers models (all-MiniLM-L6-v2, etc.)."""

    def __init__(self, model_name: str, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        chunks = []
        for start in tqdm(range(0, len(texts), batch_size), desc="SBERT encoding"):
            batch = texts[start : start + batch_size]
            emb = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=False,
            )
            chunks.append(emb.cpu())
        if not chunks:
            return torch.empty((0, self.dim), dtype=torch.float32)
        return torch.cat(chunks, dim=0)


class TransformerEncoder:
    """HuggingFace causal / encoder models (Qwen3-Embedding, etc.).

    Uses last-token pooling (suitable for decoder-based embedding models).
    """

    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16,
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        chunks = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Transformer encoding"):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            out = self.model(**inputs)
            attention_mask = inputs["attention_mask"]
            last_pos = attention_mask.sum(dim=1) - 1
            emb = out.last_hidden_state[torch.arange(len(batch), device=self.device), last_pos]
            chunks.append(emb.float().cpu())
        if not chunks:
            return torch.empty((0, self.dim), dtype=torch.float32)
        return torch.cat(chunks, dim=0)


def make_encoder(model_name: str, device: str = "cuda", max_length: int = 512):
    """Auto-detect encoder type based on model name / path."""
    name_lower = model_name.lower()
    is_sbert = any(k in name_lower for k in ("sentence-transformer", "all-minilm", "sbert"))
    if not is_sbert:
        cfg_path = Path(model_name) / "config.json"
        if cfg_path.exists():
            import json
            with open(cfg_path) as f:
                cfg = json.load(f)
            arch = cfg.get("architectures", [""])[0].lower()
            if "sentencetransformer" in arch:
                is_sbert = True
    if is_sbert:
        return SBERTEncoder(model_name, device=device)
    return TransformerEncoder(model_name, device=device, max_length=max_length)


# ---------------------------------------------------------------------------
# Data utils
# ---------------------------------------------------------------------------

def merge_title_and_abstract(titles: list[str], abstracts: list[str]) -> list[str]:
    rows: list[str] = []
    for title, abstract in zip(titles, abstracts):
        title = str(title or "").strip()
        abstract = str(abstract or "").strip()
        if title and abstract:
            rows.append(f"{title}\n\n{abstract}")
        elif title:
            rows.append(title)
        elif abstract:
            rows.append(abstract)
        else:
            rows.append("")
    return rows


def infer_tag(model_name: str) -> str:
    """Derive a short tag for output filenames from model name/path."""
    name = Path(model_name).name.lower()
    if "all-minilm" in name or "sbert" in name:
        return "sbert"
    if "qwen3-embedding" in name:
        size = ""
        for part in name.split("-"):
            if part.endswith("b"):
                size = f"_{part}"
                break
        return f"qwen3emb{size}"
    return name.replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build node and hyperedge text embeddings")
    parser.add_argument("--input-dir", type=str, default="dataset/arxiv_hg")
    parser.add_argument(
        "--model-name", type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SBERT model name/path, or HuggingFace model id/path (e.g. Qwen/Qwen3-Embedding-0.6B)",
    )
    parser.add_argument("--tag", type=str, default=None,
                        help="Output filename tag (default: auto-inferred from model name)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length (only for transformer encoder)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf-endpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    input_dir = Path(args.input_dir)
    processed_data = torch.load(input_dir / "processed_data.pt", map_location="cpu",
                                weights_only=False)

    encoder = make_encoder(args.model_name, device=args.device, max_length=args.max_length)
    tag = args.tag or infer_tag(args.model_name)

    print(f"Encoder: {type(encoder).__name__}")
    print(f"Output dim: {encoder.dim}")
    print(f"Tag: {tag}")

    node_texts = merge_title_and_abstract(processed_data["title"], processed_data["abs"])
    hyperedge_source = processed_data["hyperedge_source"].tolist()
    hyperedge_texts = [node_texts[int(source_node_id)] for source_node_id in hyperedge_source]

    print(f"Encoding {len(node_texts)} nodes ...")
    node_embeddings = encoder.encode(node_texts, args.batch_size)

    print(f"Encoding {len(hyperedge_texts)} hyperedges ...")
    hyperedge_embeddings = encoder.encode(hyperedge_texts, args.batch_size)

    emb_dir = input_dir / "embeddings" / tag
    emb_dir.mkdir(parents=True, exist_ok=True)
    node_path = emb_dir / "node_x.pt"
    he_path = emb_dir / "hyperedge_x.pt"
    torch.save(node_embeddings, node_path)
    torch.save(hyperedge_embeddings, he_path)
    print(f"Saved: {node_path}  shape={node_embeddings.shape}")
    print(f"Saved: {he_path}  shape={hyperedge_embeddings.shape}")


if __name__ == "__main__":
    main()
