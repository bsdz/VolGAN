"""
Transaction costs for the hedging backtest.

Option half-spreads are measured from the bid/ask quotes in the Kaggle AAPL option
chains. They are stored as a fraction of spot, as the median for each (call/put,
moneyness K/S, days to expiry) bucket. Using a fraction of spot handles AAPL's 2020
4:1 split: the 2016-2020 chains quote pre-split prices, while the price CSV is
split-adjusted. The table is committed as data-out/option_half_spreads.csv.
Regenerate it with:

    python transaction_costs.py data-in/aapl_2016_2020.csv data-in/aapl_2021_2023.csv

yabte charges the costs as the backtest trades (hedging_backtest.build_data and
run_backtest): the underlying pays a fixed `stock_bps` on its traded notional, and
each hedging call pays its HalfSpread column (half_spread_column) per unit traded.
Only hedge trades are charged, not the straddle itself, which is the same for every
method. Options held to expiry settle without a spread.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).parent
DEFAULT_TABLE = root_dir / "data-out/option_half_spreads.csv"
M_EDGES = (0.6, 0.8, 0.9, 0.97, 1.03, 1.1, 1.2, 1.4)  # moneyness K/S buckets
DTE_EDGES = (0, 7, 14, 21, 35)  # calendar days to expiry buckets


def estimate_half_spreads(paths):
    """Median half-spread / spot per (side, moneyness, days-to-expiry) bucket."""
    want = {"QUOTE_DATE", "UNDERLYING_LAST", "DTE", "STRIKE"}
    want |= {f"{s}_{q}" for s in "CP" for q in ("BID", "ASK")}
    frames = []
    for path in paths:
        d = pd.read_csv(
            path,
            skipinitialspace=True,
            usecols=lambda c: c.strip().strip("[]") in want,
            low_memory=False,
        )
        d.columns = [c.strip().strip("[]") for c in d.columns]
        frames.append(
            d.drop(columns="QUOTE_DATE").apply(pd.to_numeric, errors="coerce")
        )
    d = pd.concat(frames)
    d = d[(d.DTE >= 1) & (d.DTE <= DTE_EDGES[-1])]
    d["m"] = d.STRIKE / d.UNDERLYING_LAST

    rows = []
    for side, s in (("call", "C"), ("put", "P")):
        bid, ask = d[f"{s}_BID"], d[f"{s}_ASK"]
        x = d[(bid > 0) & (ask > bid)].assign(
            hs=lambda x: (x[f"{s}_ASK"] - x[f"{s}_BID"]) / 2 / x.UNDERLYING_LAST
        )
        x["mb"] = pd.cut(x.m, M_EDGES)
        x["db"] = pd.cut(x.DTE, DTE_EDGES)
        g = x.groupby(["mb", "db"], observed=False).hs
        for (mb, db), med, n in zip(g.median().index, g.median(), g.size()):
            rows.append(
                dict(
                    side=side,
                    m_lo=mb.left,
                    m_hi=mb.right,
                    dte_lo=db.left,
                    dte_hi=db.right,
                    half_spread=med,
                    n=n,
                )
            )
    return pd.DataFrame(rows)


class HalfSpreads:
    """
    Half-spread lookups in currency units per unit traded.

    stock_bps: the underlying's half-spread plus impact, in basis points of spot.
    multiplier: scales every half-spread (0 is frictionless).
    """

    def __init__(self, table, stock_bps=1.0, multiplier=1.0):
        self.stock_bps = stock_bps
        self.multiplier = multiplier
        shape = (len(M_EDGES) - 1, len(DTE_EDGES) - 1)
        self._grid = {}
        for side in ("call", "put"):
            t = table[table.side == side].sort_values(["m_lo", "dte_lo"])
            self._grid[side] = t.half_spread.to_numpy(float).reshape(shape)

    @classmethod
    def from_csv(cls, path=DEFAULT_TABLE, **kwargs):
        return cls(pd.read_csv(path), **kwargs)

    def option(self, opt, date, S):
        """Half-spread of `opt` (volgan_hedging.Option) on `date` with spot S."""
        m = np.clip(opt.strike / S, M_EDGES[0], M_EDGES[-1])
        dte = np.clip((opt.expiry - pd.Timestamp(date)).days, 1, DTE_EDGES[-1])
        i = min(np.searchsorted(M_EDGES, m, side="left") - 1, len(M_EDGES) - 2)
        j = min(np.searchsorted(DTE_EDGES, dte, side="left") - 1, len(DTE_EDGES) - 2)
        hs = self._grid["call" if opt.call else "put"][max(i, 0), max(j, 0)]
        return self.multiplier * hs * S

    @property
    def stock_cost_bps(self):
        """The underlying's cost in basis points of traded notional, for yabte's
        `cost` with `CostType.BPS`."""
        return self.multiplier * self.stock_bps


def half_spread_column(spreads, opt, spot):
    """
    `opt`'s half-spread per unit on each row of `spot` (the backtest's dates), for
    yabte's `cost_field`.

    A trade on row t fills at Open, the previous close, so it is costed with that
    close's spot and the option's moneyness and time to expiry then. On the row after
    expiry the option settles at its final price, so the spread is 0.
    """
    prev = pd.Series(spot.index, index=spot.index).shift(1)
    prev_spot = spot.shift(1)
    out = pd.Series(np.nan, index=spot.index)
    for t, d, S in zip(spot.index, prev, prev_spot):
        if pd.isna(d):
            continue
        out[t] = spreads.option(opt, d, float(S)) if d < opt.expiry else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("chains", nargs="+", type=Path, help="Kaggle option chain CSVs")
    parser.add_argument("--output", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    table = estimate_half_spreads(args.chains)
    table.to_csv(args.output, index=False, float_format="%.6g")
    print(f"Saved {len(table)} buckets to {args.output}")
    print(
        table.pivot_table(
            index=["side", "m_lo"], columns="dte_hi", values="half_spread"
        ).map(lambda v: f"{v * 1e4:.1f}bp")
    )


if __name__ == "__main__":
    main()
