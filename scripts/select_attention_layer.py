"""Choose one attention layer per model with a subject-retrieval probe.

Adapted from the layer-selection step of Turk & Neu (2026) to causal LMs.
For every nsubj / nsubj:pass dependency in UD English-EWT whose subject
precedes its head, we take the attention of the head word's first subtoken,
average it over all heads of a layer, keep only the preceding tokens and
check whether the subject's first subtoken is among the five most-attended
tokens. The layer with the highest top-5 accuracy is selected (ties: lower
mean subject rank, then lower layer).

Note: in UD the head of a copular clause is the predicate ("rusty" in "the
key is rusty"), so not every query word is a verb.

Usage (from the repository root):
    python scripts/select_attention_layer.py --max-sentences 2000
    python scripts/select_attention_layer.py --max-sentences 2000 --baselines-only

The layers used in the experiment (results/layer_selection/layer_selection_2000*.csv)
come from a run with --no-prefix-space. See the README for why this matters.
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

UD_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/"
SPLITS = ["train", "dev", "test"]
MODELS = ["gpt2", "gpt2-large", "bigscience/bloom-560m"]


def load_ud(ud_dir, splits):
    """Read UD sentences (train, dev, test in that order) as lists of
    (form, upos, head, deprel); head is 0-based, -1 for the root.
    Multi-word tokens and empty nodes are skipped. Files are downloaded if missing."""
    ud_dir = Path(ud_dir)
    ud_dir.mkdir(parents=True, exist_ok=True)
    sentences = []
    for split in splits:
        path = ud_dir / f"en_ewt-ud-{split}.conllu"
        if not path.exists():
            print(f"downloading {path.name}")
            urllib.request.urlretrieve(UD_URL + path.name, path)
        words = []
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                if words:
                    sentences.append(words)
                words = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            words.append((cols[1], cols[3], int(cols[6]) - 1, cols[7]))
        if words:
            sentences.append(words)
    return sentences


def subject_pairs(sentence):
    """(head index, subject index) for all nsubj / nsubj:pass relations."""
    return [(head, i) for i, (_, _, head, rel) in enumerate(sentence)
            if rel in ("nsubj", "nsubj:pass") and head >= 0]


def word_level_baselines(sentences):
    """Accuracy that needs no model: uniform attention and 'attend to the five
    most recent words'. Computed on words, so only approximate for tokens."""
    n_before, distance = [], []
    for s in sentences:
        for head, subj in subject_pairs(s):
            if subj < head:
                n_before.append(head)
                distance.append(head - subj)
    n_before, distance = np.array(n_before), np.array(distance)
    return {
        "pairs (subject before head)": len(n_before),
        "share with <= 5 preceding words": np.mean(n_before <= 5),
        "uniform attention, top-5 accuracy": np.mean(np.minimum(5, n_before) / n_before),
        "five most recent words, accuracy": np.mean(distance <= 5),
    }


def probe_model(model_name, sentences, prefix_space):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Only pass add_prefix_space when it is wanted, so that --no-prefix-space
    # loads the tokenizer exactly as in the original run.
    extra = {"add_prefix_space": True} if prefix_space else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **extra)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.to(device).eval()

    hits, ranks = None, None
    for s in sentences:
        enc = tokenizer([w[0] for w in s], is_split_into_words=True,
                        add_special_tokens=False, return_tensors="pt")
        word_ids = enc.word_ids(0)
        first_token = {}
        for t, w in enumerate(word_ids):
            if w is not None:
                first_token.setdefault(w, t)

        pairs = []
        for head, subj in subject_pairs(s):
            if head in first_token and subj in first_token and first_token[subj] < first_token[head]:
                pairs.append((first_token[head], first_token[subj]))
        if not pairs:
            continue

        with torch.inference_mode():
            attentions = model(input_ids=enc["input_ids"].to(device),
                               use_cache=False, output_attentions=True).attentions
        if hits is None:
            hits = [[] for _ in attentions]
            ranks = [[] for _ in attentions]

        for layer, att in enumerate(attentions):
            mean_att = att[0].float().mean(dim=0)          # average over heads
            for query, subj in pairs:
                dist = mean_att[query, :query]
                dist = dist / dist.sum().clamp_min(1e-12)
                top = torch.topk(dist, k=min(5, len(dist))).indices.tolist()
                hits[layer].append(subj in top)
                ranks[layer].append(int((dist > dist[subj]).sum()) + 1)

    return pd.DataFrame({
        "model": model_name,
        "layer": range(1, len(hits) + 1),
        "n_dependencies": [len(h) for h in hits],
        "subject_top5_accuracy": [np.mean(h) for h in hits],
        "mean_subject_rank": [np.mean(r) for r in ranks],
        "median_subject_rank": [np.median(r) for r in ranks],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--ud-dir", default="data/ud_english_ewt")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="use only the first N sentences (train comes first)")
    parser.add_argument("--no-prefix-space", action="store_true",
                        help="tokenize UD words without a leading space "
                             "(reproduces the original layer selection)")
    parser.add_argument("--baselines-only", action="store_true",
                        help="only print the model-free baselines")
    parser.add_argument("--out", default="results/layer_selection/layer_selection.csv")
    args = parser.parse_args()

    sentences = load_ud(args.ud_dir, SPLITS)[: args.max_sentences]
    print(f"{len(sentences)} sentences")
    for name, value in word_level_baselines(sentences).items():
        print(f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}")
    if args.baselines_only:
        return

    results = pd.concat([probe_model(m, sentences, not args.no_prefix_space)
                         for m in args.models], ignore_index=True)
    results.to_csv(args.out, index=False)

    best = (results.sort_values(["model", "subject_top5_accuracy", "mean_subject_rank", "layer"],
                                ascending=[True, False, True, True])
                   .groupby("model").head(1)
                   .rename(columns={"layer": "selected_layer"}))
    best_path = Path(args.out).with_name(Path(args.out).stem + "_best.csv")
    best.to_csv(best_path, index=False)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
