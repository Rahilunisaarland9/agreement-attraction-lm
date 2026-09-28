"""Statistical analysis of the six model-output files.

Measures
  surprisal     surprisal (bits) of the verb that was actually shown
  entropy       attention entropy at the verb (bits), primary attention measure
  norm_entropy  entropy / log2(context length), secondary measure

Attractor effect = plural attractor minus singular attractor, computed per item
within each subject-number x grammaticality cell. The focal attraction
contrast is the singular-subject, ungrammatical cell (e.g. "The key to the
cabinets ... were" vs "The key to the cabinet ... were"). There, attraction
predicts negative surprisal and positive entropy effects.

Careful with the plural-subject cells: there the number-mismatching attractor
is the SINGULAR one, so the sign of an attraction-like effect flips.

Outputs go to results/analysis/.
Usage (from the repository root):  python scripts/analyze_results.py
"""
from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "results" / "model_outputs"
ANALYSIS = ROOT / "results" / "analysis"

MODELS = {"gpt2": "GPT-2", "gpt2-large": "GPT-2-large", "bloom-560m": "BLOOM-560m"}
CONSTRUCTIONS = ["ORC", "PP"]
METRICS = {
    "surprisal": "target_surprisal",
    "entropy": "attention_entropy_bits",
    "norm_entropy": "attention_entropy_normalized",
}
# Diagnostics only: attention mass on the subject and on the attractor.
DIAGNOSTICS = {
    "attn_subject": "attention_to_subject",
    "attn_attractor": "attention_to_attractor",
}
# Sign of the focal effect that counts as attraction.
EXPECTED_SIGN = {"surprisal": -1, "entropy": 1, "norm_entropy": 1}
CELLS = {
    "SG_ungram": ("singular", "ungrammatical"),
    "SG_gram": ("singular", "grammatical"),
    "PL_ungram": ("plural", "ungrammatical"),
    "PL_gram": ("plural", "grammatical"),
}


def load_data():
    frames = []
    for file_key, model in MODELS.items():
        for construction in CONSTRUCTIONS:
            df = pd.read_csv(OUTPUTS / f"{file_key}_{construction.lower()}.csv")
            df["construction_name"] = construction
            df["model_name"] = model
            df["target_surprisal"] = np.where(df.grammaticality == "grammatical",
                                              df.surprisal_correct, df.surprisal_incorrect)
            # Treatment coding; reference = singular subject, singular attractor, ungrammatical.
            df["S"] = (df.subject_number == "plural").astype(int)
            df["A"] = (df.attractor_number == "plural").astype(int)
            df["G"] = (df.grammaticality == "grammatical").astype(int)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def groups(df):
    """Iterate over (construction, model, subset) in a fixed order."""
    for construction in CONSTRUCTIONS:
        for model in MODELS.values():
            yield construction, model, df[(df.construction_name == construction) & (df.model_name == model)]


def attractor_effect(g, col, subject="singular", grammaticality="ungrammatical"):
    """Per-item attractor effect (plural - singular attractor) in one cell."""
    cell = g[(g.subject_number == subject) & (g.grammaticality == grammaticality)]
    wide = cell.pivot(index="item_id", columns="attractor_number", values=col)
    return wide["plural"] - wide["singular"]


def validate(df):
    rows = []
    for construction, model, g in groups(df):
        rows.append({
            "construction": construction, "model": model,
            "rows": len(g), "items": g.item_id.nunique(),
            "conditions": g.condition.nunique(),
            "duplicates": int(g.duplicated(["item_id", "condition"]).sum()),
            "missing_values": int(g[list(METRICS.values())].isna().sum().sum()),
            "attention_layer": ",".join(map(str, sorted(g.attention_layer.unique()))),
            "heads": ",".join(map(str, sorted(g.attention_num_heads_aggregated.unique()))),
        })
    return pd.DataFrame(rows)


def condition_means(df):
    cols = {**METRICS, **DIAGNOSTICS}
    return (df.groupby(["construction_name", "model_name", "condition",
                        "subject_number", "attractor_number", "grammaticality"])
              [list(cols.values())].mean()
              .rename(columns={v: k for k, v in cols.items()})
              .reset_index())


def paired_tests(df):
    """Paired t-test of the attractor effect in every cell, for every measure."""
    rows = []
    for construction, model, g in groups(df):
        for name, col in {**METRICS, **DIAGNOSTICS}.items():
            for cell, (subject, gram) in CELLS.items():
                d = attractor_effect(g, col, subject, gram)
                t = ttest_1samp(d, 0.0)
                ci = t.confidence_interval(0.95)
                rows.append({
                    "construction": construction, "model": model, "metric": name,
                    "cell": cell, "n_items": len(d),
                    "mean_effect": d.mean(), "sd_effect": d.std(ddof=1),
                    "dz": d.mean() / d.std(ddof=1),
                    "ci95_low": ci.low, "ci95_high": ci.high,
                    "t": t.statistic, "p": t.pvalue,
                })
    out = pd.DataFrame(rows)

    # Primary family: focal cell, surprisal and entropy, 2 constructions x 3 models.
    primary = (out.cell == "SG_ungram") & out.metric.isin(["surprisal", "entropy"])
    out["p_fdr_primary"] = np.nan
    out.loc[primary, "p_fdr_primary"] = multipletests(out.loc[primary, "p"], method="fdr_bh")[1]
    return out


def mixed_models(df):
    """measure ~ S * A * G + (1 | item), one model per construction x model x measure.

    With treatment coding, A:G is the attractor x grammaticality interaction for
    SINGULAR subjects only (= SG_gram effect - SG_ungram effect), and S:A:G is
    how that interaction differs for plural subjects. A significant S:A:G is
    therefore not by itself evidence of attraction.
    """
    rows = []
    for construction, model, g in groups(df):
        g = g.assign(item=g.item_id.astype(str))
        for name, col in METRICS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(f"{col} ~ S * A * G", g, groups=g["item"]).fit(
                    reml=False, method="lbfgs", maxiter=1000, disp=False)
            for term in ["A:G", "S:A:G"]:
                rows.append({
                    "construction": construction, "model": model, "metric": name,
                    "term": term, "coef": fit.params[term], "se": fit.bse[term],
                    "z": fit.tvalues[term], "p": fit.pvalues[term],
                    "converged": fit.converged,
                })
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby(["term", "metric"]).p.transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    return out


def standardized_effects(df):
    """Focal effect in z units: each measure is z-scored over all rows of a
    model x construction. Note that this SD includes the (large) grammaticality
    effect on surprisal, which shrinks surprisal z-values."""
    rows = []
    for construction, model, g in groups(df):
        for name, col in METRICS.items():
            z = g.assign(z=(g[col] - g[col].mean()) / g[col].std(ddof=1))
            rows.append({"construction": construction, "model": model, "metric": name,
                         "focal_effect_z": attractor_effect(z, "z").mean()})
    return pd.DataFrame(rows)


def correlations(df):
    """Two kinds of cross-model correlation, per model pair:
    profile  raw values over all item x condition cells (overall similarity;
             dominated by grammaticality and lexical differences)
    focal    item-level focal attractor effects (attraction-specific)"""
    rows = []
    for construction in CONSTRUCTIONS:
        sub = df[df.construction_name == construction]
        for name, col in METRICS.items():
            wide = sub.pivot_table(index=["item_id", "condition"], columns="model_name", values=col)
            focal = {m: attractor_effect(sub[sub.model_name == m], col) for m in MODELS.values()}
            for a, b in combinations(MODELS.values(), 2):
                rows.append({"construction": construction, "metric": name, "model_a": a, "model_b": b,
                             "profile_pearson": pearsonr(wide[a], wide[b])[0],
                             "profile_spearman": spearmanr(wide[a], wide[b])[0],
                             "focal_pearson": pearsonr(focal[a], focal[b])[0]})
    return pd.DataFrame(rows)


def consistency(tests, std, corr):
    """Descriptive cross-model consistency of the focal effect (3 models only)."""
    rows = []
    focal = tests[tests.cell == "SG_ungram"]
    for construction in CONSTRUCTIONS:
        for name in METRICS:
            f = focal[(focal.construction == construction) & (focal.metric == name)]
            z = std[(std.construction == construction) & (std.metric == name)].focal_effect_z
            c = corr[(corr.construction == construction) & (corr.metric == name)]
            dz = f.dz.to_numpy()
            rows.append({
                "construction": construction, "metric": name,
                "models_expected_sign": int((np.sign(f.mean_effect) == EXPECTED_SIGN[name]).sum()),
                "models_significant_expected": int(((np.sign(f.mean_effect) == EXPECTED_SIGN[name])
                                                    & (f.p < .05)).sum()),
                "sd_z_across_models": z.std(ddof=1),
                "sd_dz_across_models": dz.std(ddof=1),
                "cv_abs_dz": np.abs(dz).std(ddof=1) / np.abs(dz).mean(),
                "mean_profile_r": c.profile_pearson.mean(),
                "mean_focal_r": c.focal_pearson.mean(),
            })
    return pd.DataFrame(rows)


def robustness(df):
    """Focal entropy effect after removing two possible artefacts:
    length    items where the plural attractor changes the number of context tokens
    subtoken  items where grammatical and ungrammatical verbs share their first
              subtoken, so the attention query is identical in both"""
    rows = []
    for construction, model, g in groups(df):
        length_change = attractor_effect(g, "attention_context_token_count")
        wide = g.pivot(index="item_id", columns="condition", values="attention_entropy_bits")
        shared = wide.index[np.isclose(wide["a"], wide["c"])]
        subsets = {"all items": length_change.index,
                   "context length unchanged": length_change.index[length_change == 0],
                   "distinct verb subtoken": length_change.index.difference(shared)}
        for name in ["entropy", "norm_entropy"]:
            effect = attractor_effect(g, METRICS[name])
            for label, keep in subsets.items():
                d = effect.loc[keep]
                rows.append({"construction": construction, "model": model, "metric": name,
                             "subset": label, "n_items": len(d), "mean_effect": d.mean(),
                             "p": ttest_1samp(d, 0).pvalue})
    return pd.DataFrame(rows)


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    df = load_data()

    val = validate(df)
    tests = paired_tests(df)
    std = standardized_effects(df)
    corr = correlations(df)
    tables = {
        "validation": val,
        "condition_means": condition_means(df),
        "paired_tests": tests,
        "mixed_models": mixed_models(df),
        "standardized_effects": std,
        "correlations": corr,
        "consistency": consistency(tests, std, corr),
        "robustness": robustness(df),
    }
    for name, table in tables.items():
        table.to_csv(ANALYSIS / f"{name}.csv", index=False)

    pd.set_option("display.width", 160)
    print(val.to_string(index=False), "\n")
    focal = tests[(tests.cell == "SG_ungram") & tests.metric.isin(METRICS)]
    print(focal[["construction", "model", "metric", "mean_effect", "dz", "p", "p_fdr_primary"]]
          .round(4).to_string(index=False), "\n")
    print(tables["consistency"].round(3).to_string(index=False))
    print(f"\nwrote {len(tables)} tables to {ANALYSIS}")


if __name__ == "__main__":
    main()
