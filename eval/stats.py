import csv
import numpy as np

JUDGE = "eval/results/judge.csv"
REF = "full_pp"
B = 10000


def load():
    d = {}
    with open(JUDGE) as f:
        for r in csv.DictReader(f):
            o = (int(r["correctness"]) + int(r["actionability"]) + int(r["depth"])) / 3
            d.setdefault(r["config"], {})[int(r["ex_id"])] = o
    return d


def ci(vals, rng):
    vals = np.asarray(vals)
    bs = vals[rng.integers(0, len(vals), (B, len(vals)))].mean(axis=1)
    return vals.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def paired(a, b, rng):
    diff = np.asarray(a) - np.asarray(b)
    bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return diff.mean(), lo, hi, p


def main():
    d = load()
    rng = np.random.default_rng(0)
    ref = d.get(REF, {})
    print(f"{'config':<20}{'n':>5}{'mean':>8}{'95% CI':>18}{'Δ vs full':>11}{'Δ 95% CI':>18}{'p':>9}")
    for c in sorted(d):
        ex = d[c]
        m, lo, hi = ci(list(ex.values()), rng)
        line = f"{c:<20}{len(ex):>5}{m:>8.3f}{f'[{lo:.3f},{hi:.3f}]':>18}"
        if c != REF and ref:
            shared = sorted(set(ex) & set(ref))
            if shared:
                a = [ref[i] for i in shared]
                b = [ex[i] for i in shared]
                dm, dlo, dhi, p = paired(a, b, rng)
                line += f"{dm:>11.3f}{f'[{dlo:.3f},{dhi:.3f}]':>18}{p:>9.4f}"
        print(line)


if __name__ == "__main__":
    main()
