import pandas as pd
from bert_score import score
from pathlib import Path

CSV = Path("eval/results/scores.csv")


def run():
    df = pd.read_csv(CSV)

    if "bertscore" in df.columns and df["bertscore"].notna().all():
        print("already done")
        return
    cands = df["gen"].fillna("").tolist()
    refs = df["ref"].fillna("").tolist()
    _, _, f1 = score(cands, refs, lang="en", model_type="roberta-large", verbose=True)

    df["bertscore"] = f1.tolist()
    df.to_csv(CSV, index=False)

    print(df.groupby("config")["bertscore"].mean())


if __name__ == "__main__":
    run()
