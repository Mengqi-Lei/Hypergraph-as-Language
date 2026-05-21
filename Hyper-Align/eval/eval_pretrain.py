import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import os
import json
import random
from tqdm import tqdm
import shortuuid
from pathlib import Path
from torch.utils.data import DataLoader

from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, get_model_name_from_path
from utils.hypergraph_features import load_hypergraph_semantic_embeddings
from utils.hypergraph_dataset_registry import (
    is_hypergraph_dataset_name,
    resolve_hypergraph_data_root,
)
from utils.hypergraph_eval import HypergraphEvalDataset, HypergraphEvalCollator
import math


def load_jsonl_slice(path: Path, start: int = -1, end: int = -1):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if start >= 0 and index < start:
                continue
            if start >= 0 and end >= 0 and index >= end:
                break
            if start < 0 and end > 0 and index >= end:
                break
            rows.append(json.loads(line))
    return rows


def normalize_eval_split(split: str) -> str:
    split_name = str(split).strip().lower()
    if split_name in {"val", "valid", "validation"}:
        return "valid"
    if split_name == "test":
        return "test"
    raise ValueError(f"Unsupported eval split: {split}. Expected one of: valid, val, test.")


def validate_eval_args(args) -> None:
    if args.eval_batch_size < 1:
        raise ValueError(f"--eval-batch-size must be >= 1, got {args.eval_batch_size}.")
    if args.eval_num_workers < 0:
        raise ValueError(f"--eval-num-workers must be >= 0, got {args.eval_num_workers}.")
    if args.flush_every < 1:
        raise ValueError(f"--flush-every must be >= 1, got {args.flush_every}.")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError(f"--max-new-tokens must be >= 1, got {args.max_new_tokens}.")
    if args.random_sample_size is not None and args.random_sample_size < 1:
        raise ValueError(f"--random-sample-size must be >= 1, got {args.random_sample_size}.")


def strip_stop_string(text: str, stop_str: str) -> str:
    text = text.strip()
    if stop_str and text.endswith(stop_str):
        text = text[:-len(stop_str)]
    return text.strip()


def maybe_random_sample_entries(entries, sample_size: int | None, seed: int, label: str):
    if sample_size is None:
        return entries
    total = len(entries)
    if sample_size >= total:
        print(
            f"Random sampling skipped for {label}: requested {sample_size}, "
            f"but only {total} entries are available."
        )
        return entries

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(total), sample_size))
    sampled_entries = [entries[index] for index in selected_indices]
    print(
        f"Randomly sampled {len(sampled_entries)} / {total} {label} "
        f"with seed={seed}."
    )
    return sampled_entries


def resolve_hypergraph_prompt_file(
    hyper_data_root: Path,
    task: str,
    max_incident_hyperedges: int,
    max_members_per_hyperedge: int,
    split: str,
) -> Path:
    normalized_split = normalize_eval_split(split)
    if task == "nc":
        prefix = f"node_task_{task}_hg_{max_incident_hyperedges}_{max_members_per_hyperedge}_{normalized_split}"
    else:
        prefix = f"he_task_{task}_hg_{max_incident_hyperedges}_{max_members_per_hyperedge}_{normalized_split}"

    for parent in [hyper_data_root / "samples", hyper_data_root]:
        prebaked = parent / f"{prefix}_prebaked.jsonl"
        if prebaked.exists():
            return prebaked

    raise FileNotFoundError(
        "Cannot find the required prebaked hypergraph evaluation jsonl:\n"
        f"  {hyper_data_root / 'samples' / (prefix + '_prebaked.jsonl')}\n"
        "Run scripts/prebake_data.py first to generate prebaked data."
    )


def resolve_overview_semantics_file(
    hyper_data_root: Path,
    task: str,
    max_incident_hyperedges: int,
    max_members_per_hyperedge: int,
    split: str,
    pretrained_embedding_type: str = "sbert",
) -> Path | None:
    normalized_split = normalize_eval_split(split)
    prefix = "node" if task == "nc" else "he"
    path = (hyper_data_root / "overview" / pretrained_embedding_type
            / f"{prefix}_{max_incident_hyperedges}_{max_members_per_hyperedge}_{normalized_split}.pt")
    return path if path.exists() else None


def _eval_hypergraph_worker(rank, world_size, questions_chunk, args, output_path):
    """Single-GPU worker: loads model onto cuda:{rank}, runs inference on its
    chunk of questions, and writes results to *output_path*."""
    disable_torch_init()

    eval_device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if rank == 0:
        print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        cache_dir=args.cache_dir,
        device_map={"": rank} if torch.cuda.is_available() else "cpu",
        dtype=args.eval_dtype,
    )
    eval_dtype_t = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16,  "float16":  torch.float16,
        "fp32": torch.float32,  "float32":  torch.float32,
    }[args.eval_dtype]
    if hasattr(model, "hf_device_map"):
        model = model.eval()
    else:
        model = model.to(eval_dtype_t).to(eval_device).eval()

    hyper_data_root = Path(args.hyper_data_root)
    processed_data = torch.load(hyper_data_root / "processed_data.pt", map_location="cpu", weights_only=False)

    node_embeddings, hyperedge_embeddings = load_hypergraph_semantic_embeddings(
        hyper_data_root=str(hyper_data_root),
        node_embedding_path=args.node_embedding_path,
        hyperedge_embedding_path=args.hyperedge_embedding_path,
        pretrained_embedding_type=args.pretrained_embedding_type,
        dtype=torch.float16,
    )

    overview_semantics: dict = {}
    ov_path = resolve_overview_semantics_file(
        hyper_data_root, args.task,
        args.max_incident_hyperedges, args.max_members_per_hyperedge,
        args.split,
        pretrained_embedding_type=getattr(args, "pretrained_embedding_type", "sbert"),
    )
    if ov_path is not None:
        overview_semantics = torch.load(ov_path, map_location="cpu")
        if rank == 0:
            print(f"Loaded overview semantics: {ov_path} ({len(overview_semantics)} entries)")

    do_sample = args.temperature > 0
    max_new_tokens = 128 if args.max_new_tokens is None else args.max_new_tokens
    stop_conv = conv_templates[args.conv_mode].copy()
    stop_str = stop_conv.sep if stop_conv.sep_style != SeparatorStyle.TWO else stop_conv.sep2

    center_kind = "vertex" if args.task == "nc" else "hyperedge"

    dataset = HypergraphEvalDataset(
        rows=questions_chunk,
        task=args.task,
        tokenizer=tokenizer,
        conv_mode=args.conv_mode,
        processed_data=processed_data,
        node_embeddings=node_embeddings,
        hyperedge_embeddings=hyperedge_embeddings,
        overview_semantics=overview_semantics,
        max_hypergraph_tokens=args.max_hypergraph_tokens,
        projector_incidence_mode=getattr(args, "projector_incidence_mode", "sample_real"),
        hidt_center_kind=center_kind,
        hidt_max_depth=args.formal_hidt_depth,
        hidt_max_incident_hyperedges=args.max_incident_hyperedges,
        hidt_max_members_per_hyperedge=args.max_members_per_hyperedge,
        hidt_max_child_hyperedges=args.max_child_hyperedges,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.eval_num_workers > 0,
        collate_fn=HypergraphEvalCollator(),
    )
    total_batches = math.ceil(len(dataset) / args.eval_batch_size) if len(dataset) > 0 else 0
    if rank == 0:
        print(
            f"Hypergraph eval config: batch_size={args.eval_batch_size}, "
            f"num_workers={args.eval_num_workers}, max_new_tokens={max_new_tokens}, "
            f"flush_every={args.flush_every}, world_size={world_size}"
        )

    written_count = 0
    with open(output_path, "w", encoding="utf-8") as ans_file:
        iterator = tqdm(dataloader, total=total_batches, desc=f"GPU {rank}") if rank == 0 else dataloader
        for batch in iterator:
            input_ids = batch["input_ids"].to(eval_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(eval_device, non_blocking=True)
            graph = batch["graph"].to(eval_device, non_blocking=True)
            graph_emb = batch["graph_emb"].to(eval_device, non_blocking=True)
            graph_aux = {
                key: value.to(eval_device, non_blocking=True)
                for key, value in batch["graph_aux"].items()
            }

            gen_kwargs = dict(
                attention_mask=attention_mask,
                graph_emb=graph_emb,
                graph=graph,
                graph_aux=graph_aux,
                do_sample=do_sample,
                num_beams=args.num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            if do_sample:
                gen_kwargs["temperature"] = args.temperature
                if args.top_p is not None:
                    gen_kwargs["top_p"] = args.top_p
            else:
                gen_kwargs["temperature"] = None
                gen_kwargs["top_p"] = None
                gen_kwargs["top_k"] = None

            with torch.inference_mode():
                output_ids = model.generate(input_ids, **gen_kwargs)

            input_token_len = input_ids.shape[1]
            decoded_outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:],
                skip_special_tokens=True,
            )
            for question_id, prompt_text, gt_text, raw_output in zip(
                batch["question_ids"],
                batch["prompts"],
                batch["gts"],
                decoded_outputs,
            ):
                outputs = strip_stop_string(raw_output, stop_str)
                ans_file.write(
                    json.dumps(
                        {
                            "question_id": int(question_id),
                            "prompt": prompt_text,
                            "text": outputs,
                            "gt": gt_text,
                            "answer_id": shortuuid.uuid(),
                        }
                    )
                    + "\n"
                )
                written_count += 1
                if written_count % args.flush_every == 0:
                    ans_file.flush()
        ans_file.flush()
    if rank == 0:
        print(f"GPU {rank} finished: wrote {written_count} results to {output_path}")


def _spawn_worker(rank, world_size, all_chunks, args, tmp_files):
    """Entry point for torch.multiprocessing.spawn (rank is prepended automatically)."""
    _eval_hypergraph_worker(rank, world_size, all_chunks[rank], args, tmp_files[rank])


def _merge_result_files(tmp_files, final_path):
    """Merge per-GPU result files into a single file sorted by question_id."""
    all_lines = []
    for tmp_path in tmp_files:
        if not os.path.exists(tmp_path):
            continue
        with open(tmp_path, "r", encoding="utf-8") as handle:
            for line in handle:
                all_lines.append(json.loads(line))
    all_lines.sort(key=lambda row: int(row["question_id"]))
    with open(final_path, "w", encoding="utf-8") as handle:
        for row in all_lines:
            handle.write(json.dumps(row) + "\n")
    for tmp_path in tmp_files:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print(f"Merged {len(all_lines)} results into {final_path}")


def eval_hypergraph_model(args):
    if args.task not in {"nc", "hecls"}:
        raise ValueError(
            "eval_pretrain.py currently supports hypergraph datasets with tasks from nc/hecls."
        )

    hyper_data_root = Path(args.hyper_data_root)
    processed_data = torch.load(hyper_data_root / "processed_data.pt", map_location="cpu", weights_only=False)
    declared_tasks = processed_data.get("supported_tasks")
    if declared_tasks is not None and args.task not in {str(task).strip() for task in declared_tasks}:
        raise ValueError(
            f"Dataset at {hyper_data_root} supports only {list(declared_tasks)}, but got task={args.task}."
        )
    eval_split = normalize_eval_split(args.split)
    prompt_file = (
        Path(args.test_path)
        if args.test_path is not None
        else resolve_hypergraph_prompt_file(
            hyper_data_root=hyper_data_root,
            task=args.task,
            max_incident_hyperedges=args.max_incident_hyperedges,
            max_members_per_hyperedge=args.max_members_per_hyperedge,
            split=eval_split,
        )
    )
    selected_split = "custom" if args.test_path is not None else eval_split
    print(f"Load from {prompt_file} (split={selected_split})\n")
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    questions = load_jsonl_slice(prompt_file, start=args.start, end=args.end)
    questions = maybe_random_sample_entries(
        questions,
        sample_size=args.random_sample_size,
        seed=args.random_sample_seed,
        label="hypergraph eval samples",
    )

    num_gpus = args.num_gpus if args.num_gpus > 0 else torch.cuda.device_count()
    num_gpus = max(num_gpus, 1)
    print(f"Total questions: {len(questions)}, using {num_gpus} GPU(s)")

    if num_gpus <= 1:
        _eval_hypergraph_worker(0, 1, questions, args, answers_file)
        return

    chunks = [questions[i::num_gpus] for i in range(num_gpus)]
    tmp_files = [f"{answers_file}.rank{i}" for i in range(num_gpus)]

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        _spawn_worker,
        args=(num_gpus, chunks, args, tmp_files),
        nprocs=num_gpus,
        join=True,
    )

    _merge_result_files(tmp_files, answers_file)

def eval_model(args):
    validate_eval_args(args)
    args.hyper_data_root = resolve_hypergraph_data_root(args.dataset, args.hyper_data_root)
    return eval_hypergraph_model(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/path/to/Hyper-Align-checkpoint")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv_hg")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--hyper_data_root", type=str, default="../HyperAlign-Bench/dataset/arxiv_hg")
    parser.add_argument("--hyper_template", type=str, default="HIDT_O")
    parser.add_argument("--node_embedding_path", type=str, default=None)
    parser.add_argument("--hyperedge_embedding_path", type=str, default=None)
    parser.add_argument("--max_hypergraph_tokens", type=int, default=160)
    parser.add_argument("--max_incident_hyperedges", type=int, default=8)
    parser.add_argument("--max_members_per_hyperedge", type=int, default=8)
    parser.add_argument("--max_child_hyperedges", type=int, default=1)
    parser.add_argument("--formal_hidt_depth", type=int, default=3)
    parser.add_argument("--overview_hops", type=int, default=2)
    parser.add_argument("--overview_order_buckets", type=int, default=4)
    parser.add_argument("--projector_incidence_mode", type=str, default="sample_real")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--random-sample-size", type=int, default=None)
    parser.add_argument("--random-sample-seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs for parallel eval. 0 = auto-detect all available GPUs.")
    parser.add_argument(
        "--eval-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help=(
            "Inference dtype for the LLM + projector. Defaults to bf16 to match "
            "training. Use fp16 only for comparison runs; HECLS in "
            "particular suffers severe NaN-logits collapse under fp16."
        ),
    )
    args = parser.parse_args()

    eval_model(args)
