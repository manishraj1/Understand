"""
p1_stratify.py — Difficulty stratification pilot (Document B §5.2).

K=16 samples per candidate prompt at T=0.8; empirical pass rate -> strata:
easy >= 0.90, medium in [0.40, 0.70], hard <= 0.20. Gaps between strata are
deliberate (buffer zones dropped). Selects 50 prompts/stratum (or fewer with
a loud warning). Seeds burned.

Data: expects a JSONL of {"id":..., "question":..., "answer":...}.
  --data gsm8k_test.jsonl        (user supplies on rig; loader for HF datasets
                                  included behind --hf-dataset gsm8k)
Smoke: python p1_stratify.py --backend mock --synthetic 30 --k 4 --outdir out
"""
import argparse, json, random
from harness import (GenConfig, make_backend, RunLog, hard_assert, burn_seeds,
                     extract_final_answer, normalize_numeric, thinking_tokens,
                     verify_summary)
from prompts import build_exp1_prompt

STRATA = {"easy": (0.90, 1.01), "medium": (0.40, 0.701), "hard": (-0.01, 0.201)}


def load_items(a):
    if a.synthetic:
        rng = random.Random(a.seed)
        return [{"id": f"syn{i}", "question": f"Compute {i} + {i+11}. SYNTH",
                 "answer": "18"} for i in range(a.synthetic)]
    if a.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(a.hf_dataset, "main", split="test")
        items = []
        for i, r in enumerate(ds):
            ans = r["answer"].split("####")[-1].strip() if "####" in r["answer"] else r["answer"]
            items.append({"id": f"{a.hf_dataset}-{i}", "question": r["question"],
                          "answer": ans})
        return items
    hard_assert(a.data is not None, "provide --data / --hf-dataset / --synthetic")
    return [json.loads(l) for l in open(a.data)]


def recompute(recs):
    n = len(recs)
    pass_rates = [r["pass_rate"] for r in recs]
    return {"n_items": n,
            "mean_pass": round(sum(pass_rates) / max(n, 1), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--data"); ap.add_argument("--hf-dataset")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--per-stratum", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()

    cfg = GenConfig(model_id=a.model, backend=a.backend, seed=a.seed)
    be = make_backend(cfg)
    burn_seeds("p1_stratify", [a.seed], a.outdir)
    items = load_items(a)
    log = RunLog(a.outdir, f"p1_stratify_{a.backend}", cfg,
                 {"k": a.k, "n_items": len(items)})

    prompts = [build_exp1_prompt("baseline", it["question"])[0] for it in items]
    outs = be.generate(prompts, n=a.k)
    mem = []
    for it, os_ in zip(items, outs):
        gold = normalize_numeric(it["answer"])
        passes, lens = [], []
        for o in os_:
            pred = normalize_numeric(extract_final_answer(o))
            passes.append(int(pred is not None and pred == gold))
            lens.append(thinking_tokens(o))
        pr = sum(passes) / len(passes)
        stratum = next((s for s, (lo, hi) in STRATA.items() if lo <= pr < hi), None)
        rec = {"id": it["id"], "pass_rate": pr, "stratum": stratum,
               "mean_think_len": sum(lens) / len(lens), "k": a.k}
        mem.append(rec); log.write(dict(rec))

    summary = recompute(mem)
    # stratum selection
    rng = random.Random(a.seed + 1)
    selection = {}
    for s in STRATA:
        pool = [r["id"] for r in mem if r["stratum"] == s]
        rng.shuffle(pool)
        selection[s] = pool[: a.per_stratum]
        if len(selection[s]) < a.per_stratum:
            print(f"[WARN] stratum '{s}': only {len(selection[s])}/"
                  f"{a.per_stratum} items — enlarge candidate pool")
    summary["selected_counts"] = {s: len(v) for s, v in selection.items()}
    json.dump(selection, open(f"{a.outdir}/p1_selection.json", "w"), indent=2)
    log.close(summary)
    verify_summary(log.path, lambda r: recompute(r))
    print("P1 strata counts:", summary["selected_counts"])


if __name__ == "__main__":
    main()
