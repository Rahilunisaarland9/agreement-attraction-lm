"""Compute verb surprisal and attention entropy for one model and one stimulus file.

For every sentence prefix (e.g. "The key to the cabinets unsurprisingly") we compute

  * the surprisal of the correct and the incorrect verb form, summed over
    subword tokens, and
  * attention statistics at the target verb: we take the attention of the
    verb's FIRST subtoken at one layer, keep only the preceding tokens,
    renormalize, and compute Shannon entropy (bits) for every head. The
    reported value is the mean over heads. We also report the attention
    mass on the subject and on the attractor noun.

The layer per model was chosen beforehand with select_attention_layer.py
(see LAYERS below and the README).

Usage (from the repository root):
    python scripts/run_experiment.py --model gpt2 \
        --data data/orc_stimuli.csv --out results/model_outputs/gpt2_orc.csv
"""
import argparse
import math

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Layers are 1-based here (layer 1 = first transformer block).
LAYERS = {
    "gpt2": 5,
    "gpt2-large": 16,
    "bigscience/bloom-560m": 11,
}

REQUIRED_COLUMNS = [
    "item_id", "condition", "subject_number", "attractor_number",
    "grammaticality", "prefix", "target_verb", "correct_verb",
    "incorrect_verb", "subject", "attractor",
]


def continuation_logprob(model, tokenizer, prefix, word, device):
    """Return log P(word | prefix), summed over the word's subtokens, and the
    number of subtokens. No BOS token is added."""
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    word_ids = tokenizer(" " + word, add_special_tokens=False).input_ids
    ids = torch.tensor([prefix_ids + word_ids], device=device)

    with torch.inference_mode():
        log_probs = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0], dim=-1)

    # The token at position t is predicted by the logits at position t - 1.
    start = len(prefix_ids)
    total = sum(
        float(log_probs[start + j - 1, tok]) for j, tok in enumerate(word_ids)
    )
    return total, len(word_ids)


def char_span(text, phrase):
    """Character span of `phrase` in `text`; the phrase must occur exactly once."""
    start = text.find(phrase)
    if start < 0 or text.find(phrase, start + 1) >= 0:
        raise ValueError(f"{phrase!r} must occur exactly once in {text!r}")
    return start, start + len(phrase)


def tokens_in_span(offsets, start, end):
    """Indices of tokens whose character offsets overlap [start, end)."""
    return [i for i, (s, e) in enumerate(offsets) if e > start and s < end]


def phrase_tokens(tokenizer, prefix, phrase):
    """Token indices of a noun (subject or attractor) within the prefix."""
    offsets = tokenizer(prefix, add_special_tokens=False,
                        return_offsets_mapping=True)["offset_mapping"]
    return tokens_in_span(offsets, *char_span(prefix, phrase))


def verb_attention(model, tokenizer, prefix, verb, layer, device):
    """Attention of the verb's first subtoken over the preceding tokens.

    Returns a [heads, n_context] tensor (each row sums to 1), the index of the
    query token and the number of verb subtokens.
    """
    text = prefix + " " + verb
    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    verb_tokens = tokens_in_span(offsets, len(prefix) + 1, len(text))
    query = verb_tokens[0]

    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"].to(device),
                    use_cache=False, output_attentions=True)

    # out.attentions[layer - 1] has shape [batch, heads, query, key].
    att = out.attentions[layer - 1][0, :, query, :query].float()
    # Causal attention also puts some weight on the query token itself;
    # we drop it and renormalize over the preceding context.
    att = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return att, query, len(verb_tokens)


def entropy_bits(p):
    p = p.clamp_min(1e-12)
    return -(p * torch.log2(p)).sum(dim=-1)


def attention_features(model, tokenizer, row, layer, device):
    prefix = str(row["prefix"])
    att, query, n_verb_tokens = verb_attention(
        model, tokenizer, prefix, str(row["target_verb"]), layer, device)

    n_context = att.shape[-1]
    h = entropy_bits(att)  # one value per head
    # Normalized entropy: divide by the maximum possible entropy, log2(n_context).
    h_norm = h / math.log2(n_context) if n_context > 1 else torch.zeros_like(h)

    subj = phrase_tokens(tokenizer, prefix, str(row["subject"]))
    attr = phrase_tokens(tokenizer, prefix, str(row["attractor"]))

    # Everything is computed per head and then averaged over heads.
    return {
        "attention_layer": layer,
        "attention_entropy_bits": float(h.mean()),
        "attention_entropy_normalized": float(h_norm.mean()),
        "attention_to_subject": float(att[:, subj].sum(dim=-1).mean()),
        "attention_to_attractor": float(att[:, attr].sum(dim=-1).mean()),
        "attention_num_heads_aggregated": att.shape[0],
        "target_token_count": n_verb_tokens,
        "target_query_token_index": query,
        "attention_context_token_count": n_context,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="gpt2",
                        help="Hugging Face model name (see LAYERS)")
    parser.add_argument("--data", required=True, help="stimulus CSV")
    parser.add_argument("--out", required=True, help="output CSV")
    parser.add_argument("--layer", type=int, default=None,
                        help="override the attention layer (1-based)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N rows (quick test)")
    args = parser.parse_args()

    layer = args.layer or LAYERS.get(args.model)
    if layer is None:
        parser.error(f"no layer defined for {args.model}; pass --layer")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Eager attention is needed to get the attention weights back.
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")
    model.to(device).eval()
    print(f"{args.model} on {device}, attention layer {layer}")

    stimuli = pd.read_csv(args.data)
    missing = [c for c in REQUIRED_COLUMNS if c not in stimuli.columns]
    if missing:
        raise ValueError(f"missing columns in {args.data}: {missing}")
    if args.limit:
        stimuli = stimuli.head(args.limit)

    rows = []
    for _, r in tqdm(stimuli.iterrows(), total=len(stimuli)):
        prefix = str(r["prefix"])
        lp_correct, n_correct = continuation_logprob(model, tokenizer, prefix, str(r["correct_verb"]), device)
        lp_incorrect, n_incorrect = continuation_logprob(model, tokenizer, prefix, str(r["incorrect_verb"]), device)

        rows.append({
            "model": args.model,
            "item_id": r["item_id"],
            "construction": r.get("construction", ""),
            "condition": r["condition"],
            "subject_number": r["subject_number"],
            "attractor_number": r["attractor_number"],
            "grammaticality": r["grammaticality"],
            "prefix": prefix,
            "target_verb": r["target_verb"],
            "correct_verb": r["correct_verb"],
            "incorrect_verb": r["incorrect_verb"],
            "correct_num_tokens": n_correct,
            "incorrect_num_tokens": n_incorrect,
            "logprob_correct": lp_correct,
            "logprob_incorrect": lp_incorrect,
            "logprob_difference": lp_correct - lp_incorrect,
            "surprisal_correct": -lp_correct,
            "surprisal_incorrect": -lp_incorrect,
            **attention_features(model, tokenizer, r, layer, device),
        })

    out = pd.DataFrame(rows).sort_values(["item_id", "condition"])
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
