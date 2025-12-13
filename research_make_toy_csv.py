from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from utils_core import set_global_seed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create a deterministic toy CSV dataset for Strategy-Miner CSV loader.")
    p.add_argument("--out", default="toy_data.csv", help="Output CSV path (default: toy_data.csv)")
    p.add_argument("--rows", type=int, default=800, help="Number of rows")
    p.add_argument("--features", type=int, default=6, help="Number of feature columns")
    p.add_argument("--freq", default="D", help="Pandas frequency (default: D)")
    p.add_argument("--start", default="2016-01-01", help="Start date")
    p.add_argument("--seed", type=int, default=42, help="Seed")
    args = p.parse_args(argv)

    n = int(args.rows)
    k = int(args.features)
    set_global_seed(int(args.seed))

    idx = pd.date_range(str(args.start), periods=n, freq=str(args.freq))

    latent = np.random.normal(size=(n, 1))
    feats = 0.7 * np.random.normal(size=(n, k)) + 0.3 * latent

    w_true = np.random.normal(size=(k,))
    w_true = w_true / (np.linalg.norm(w_true) + 1e-12)

    alpha = 0.0008
    noise = np.random.normal(scale=0.01, size=(n,))
    pred = np.zeros(n)
    pred[1:] = feats[:-1].dot(w_true)
    rets = alpha * pred + noise
    price = 100.0 * np.cumprod(1.0 + rets)

    df = pd.DataFrame()
    df["timestamp"] = idx
    df["price"] = price
    for j in range(k):
        df[f"feat{j}"] = feats[:, j]

    df.to_csv(str(args.out), index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
