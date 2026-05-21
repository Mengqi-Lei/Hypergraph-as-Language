import sys
sys.path.append("./")
sys.path.append("./utils")

import torch
import json
import argparse
from sklearn.metrics import roc_auc_score

from utils.hypergraph_dataset_registry import (
    is_hypergraph_dataset_name,
    resolve_hypergraph_data_root,
)


def eval_arxiv_nc(res_path):
    data=torch.load("dataset/ogbn-arxiv/processed_data.pt")
    labels=data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    overall_correct=0
    strict_correct=0
    error=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            short_label = short_labels[y]
            label=labels[y]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error.append((ans, label))
            if args.sample > 0 and all_sample >= args.sample:
                break
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")


def eval_lp(res_path):
    all_sample=0
    correct = 0
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            ans = res["text"].strip()
            label=res["gt"].strip()
            all_sample += 1
            if ("yes" in ans and "yes" in label) or ("yes" not in ans and "no" in label):
                correct += 1
            if args.sample > 0 and all_sample >=  args.sample:
                break
    acc = correct / all_sample
    print(f"Test samples: {all_sample}\ncorrect: {correct}\n acc: {acc:.4f}")

def eval_lprank(res_path):
    all_sample=0
    correct = 0
    y_true = []
    y_pred=[]
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            logit = res["logit"]
            score = torch.softmax(torch.tensor(logit[:2]), dim=-1)[0].item()
            label=res["gt"].strip()
            if label == "yes":
                y_true.append(1)
            else:
                y_true.append(0)
            y_pred.append(score)
    auc = roc_auc_score(y_true, y_pred)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    acc = ((y_pred>0.5)==y_true).sum()/y_pred.shape[0]

    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    y_pos=y_pred[y_true==1]
    y_neg=y_pred[y_true==0]
    y_neg_sort, _ = torch.sort(y_neg)
    for n in [10,50,100,200,500,1000]:
        if n > y_neg_sort.shape[0]:
            break
        th = y_neg_sort[-n]
        h = (y_pos>th).sum()/y_pos.shape[0]
        print(f"Hits@{n}: {h:.4f}")


def eval_arxiv_hg_nc(res_path):
    data = torch.load(args.hyper_data_root + "/processed_data.pt", map_location="cpu", weights_only=False)
    labels = data["label_texts"]
    short_labels = [label.split("(")[0].strip().lower() for label in labels]
    ys = data["y"].tolist()

    all_sample = 0
    strict_correct = 0
    overall_correct = 0
    with open(res_path, "r", encoding="utf-8") as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"].strip().lower()
            label = labels[ys[res["question_id"]]].strip().lower()
            short_label = short_labels[ys[res["question_id"]]]
            if ans == label:
                strict_correct += 1
                overall_correct += 1
            elif short_label in ans and sum(short in ans for short in short_labels) == 1:
                overall_correct += 1
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(
        f"Test samples: {all_sample}\n"
        f"strict_acc: {strict_correct / all_sample:.4f}\n"
        f"overall_acc: {overall_correct / all_sample:.4f}"
    )


def eval_arxiv_hg_hecls(res_path):
    data = torch.load(args.hyper_data_root + "/processed_data.pt", map_location="cpu", weights_only=False)
    labels = data["label_texts"]
    short_labels = [label.split("(")[0].strip().lower() for label in labels]
    ys = data["y"].tolist()
    hyperedge_source = data["hyperedge_source"].tolist()

    all_sample = 0
    strict_correct = 0
    overall_correct = 0
    with open(res_path, "r", encoding="utf-8") as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"].strip().lower()
            source_node_id = hyperedge_source[res["question_id"]]
            label = labels[ys[source_node_id]].strip().lower()
            short_label = short_labels[ys[source_node_id]]
            if ans == label:
                strict_correct += 1
                overall_correct += 1
            elif short_label in ans and sum(short in ans for short in short_labels) == 1:
                overall_correct += 1
            if args.sample > 0 and all_sample >= args.sample:
                break
    if all_sample == 0:
        print("Test samples: 0\nstrict_acc: 0.0000\noverall_acc: 0.0000")
        return
    print(
        f"Test samples: {all_sample}\n"
        f"strict_acc: {strict_correct / all_sample:.4f}\n"
        f"overall_acc: {overall_correct / all_sample:.4f}"
    )


def eval_products_nc(res_path):
    eval_set = set()
    data=torch.load("dataset/ogbn-products/processed_data.pt")
    labels=data.label_names
    ys=data.y.numpy().tolist()

    all_sample=0
    strict_correct=0
    overall_correct=0
    with open(res_path, 'r') as f:
        for line in f:
            if args.sample > 0 and all_sample >= args.sample:
                break
            all_sample+=1
            res = json.loads(line)
            if res['question_id'] in eval_set:
                print(f"{res['question_id']} repeat!!")
                return
            eval_set.add(res['question_id'])
            ans = res["text"].strip()
            y=ys[res["question_id"]][0]
            label=labels[y].strip()
            if label.lower()==ans.lower():
                strict_correct+=1
                overall_correct+=1
            elif label.lower() in ans.lower() and sum([l.lower() in ans.lower() for l in labels])<=2:
                overall_correct += 1

    if all_sample == 0:
        print("Test samples: 0\nstrict_acc: 0.0000\noverall_acc: 0.0000")
        return
    overall_acc = overall_correct / all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")

def eval_pubmed_nc(res_path):
    data=torch.load("dataset/pubmed/processed_data.pt")
    labels=data.label_texts
    short_labels = [l[18:] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    strict_correct=0
    overall_correct=0
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            short_label = short_labels[y]
            label=labels[y]
            if ans.lower().strip() == label.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower().strip() in ans.lower().strip() and sum([la.lower().strip() in ans.lower().strip() for la in short_labels]) == 1:
                overall_correct += 1
            if args.sample > 0 and all_sample >= args.sample:
                break

    overall_acc = overall_correct / all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")


def eval_cora_nc(res_path):
    data=torch.load("dataset/cora/processed_data.pt")
    labels=data.label_texts
    short_labels = [l.split('_')[0] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    correct=0
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            label=labels[y]
            short_label=short_labels[y]
            if short_label.strip().lower() in ans.strip().lower() and sum([l.strip().lower() in ans.strip().lower() for l in short_labels])==1:
                correct+=1
    acc=correct/all_sample
    print(f"Test samples: {all_sample}\nacc: {acc:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="./results/answers.jsonl")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--hyper_data_root", type=str, default="../HyperAlign-Bench/dataset/arxiv_hg")
    args = parser.parse_args()

    func_dict = {
        "arxiv":{
            "nc": eval_arxiv_nc,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "products": {
            "nc": eval_products_nc,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "pubmed": {
            "nc": eval_pubmed_nc,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "cora": {
            "nc": eval_cora_nc,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "arxiv_hg": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "ogbn-arxiv-hg": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "cora_cc": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "cora_co_hg": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "pubmed": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "pubmed_hg": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "dblp": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "dblp_a_hg": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "imdb": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
        "imdb_hg": {
            "nc": eval_arxiv_hg_nc,
            "hecls": eval_arxiv_hg_hecls,
        },
    }

    if is_hypergraph_dataset_name(args.dataset):
        args.hyper_data_root = resolve_hypergraph_data_root(args.dataset, args.hyper_data_root)
    func=func_dict[args.dataset][args.task]
    func(args.res_path)
