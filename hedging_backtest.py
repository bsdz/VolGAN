"""
Backtest VolGAN scenario-regression hedging against Black-Scholes delta and
delta-vega hedging with yabte (https://github.com/bsdz/yabte).

Set-up, following Vuletić & Cont (2025), Section 5.5: back-to-back one-month
straddles (a call and a put at strike_mult x spot, 1.2 by default as in the paper),
each held to expiry and hedged
daily at the close. Options are marked off the real smoothed surface each day, with
Black-Scholes at r = 0. Each method trades in its own book, which is long the
straddle and short the hedge, so the book's value is the hedging tracking error Z_t.

Hedge trades pay half the bid/ask spread measured from the Kaggle option chains
(see transaction_costs.py), charged by yabte as they trade; --cost-mult 0 gives the
paper's frictionless set-up.

Example:
    python hedging_backtest.py --model data-out/volgan_res.pkl --device cpu
"""

import argparse
import warnings
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from yabte.backtest import (
    Book,
    CostType,
    OHLCAsset,
    PositionalOrder,
    Strategy,
    StrategyRunner,
)

from VolGAN import load_results
from volgan_hedging import (
    METHODS,
    VOLGAN_METHODS,
    Market,
    VolGANScenarios,
    hedge_ratios,
    make_rolls,
    option_prices,
    position_metrics,
)
from transaction_costs import DEFAULT_TABLE, HalfSpreads, half_spread_column

root_dir = Path(__file__).parent
SPOT = "SPOT"


@dataclass(kw_only=True)
class HedgeStrategy(Strategy):
    """
    Holds `quantity` straddles of the live roll and is short its hedge.

    on_open of row t sees only the previous close (Open fields), so decisions use
    date t-1's spot and surface and the orders fill at t-1's closing prices.

    Each rebalance records (Strategy.record) the vols, Black-Scholes greeks, hedge
    ratios and the hedged book's net delta/vega (see position_metrics), with `date`
    the close traded at; metrics_frame() collects them.

    Orders are PositionalOrders for the target holding that trade only the
    difference, so the strategy never reads its book and runs on either engine.
    """

    method: str
    book: str

    def on_open(self):
        data = self.data
        if len(data) < 2:
            return
        p = self.params
        date = data.index[-2]  # the close we are trading at

        # Order only assets with a price to fill at: an option has no Open before it
        # is bought or after the row that sells it, and its target is 0 then anyway.
        opens = data.iloc[-1].xs("Open", level=1)
        targets = {name: Decimal(0) for name in opens.index[opens.notna()]}
        for roll in p.rolls:
            if roll.entry <= date < roll.expiry:
                for leg in roll.legs:
                    targets[leg.name] = Decimal(str(p.quantity))
                phi = hedge_ratios(
                    self.method,
                    p.market,
                    date,
                    roll,
                    p.quantity,
                    p.scenarios,
                    p.max_hedge_ratio,
                )
                self.record(
                    date=date, **position_metrics(p.market, date, roll, p.quantity, phi)
                )
                targets[SPOT] = -Decimal(str(round(phi["spot"], 6)))
                targets[roll.hedge_call(date).name] = -Decimal(
                    str(round(phi["hedge"], 6))
                )

        for name, target in targets.items():
            self.orders.append(
                PositionalOrder(
                    asset_name=name, size=target, book=self.book, close_first=False
                )
            )


def hedging_calls(rolls):
    """{name: Option} of every roll's hedging calls (the entry call and re-strikes)."""
    calls = {}
    for roll in rolls:
        calls[roll.atm_call.name] = roll.atm_call
        calls.update({o.name: o for o in roll.hedges.values()})
    return calls


def build_data(market, rolls, start, end, spreads=None):
    """
    (asset, field) frame with Open = previous close, from start to end.

    Expiry relies on this shift: an option's prices are NaN after its expiry, so
    HedgeStrategy closes the position on the row after expiry, filling at Open (the
    expiry-day close), before that row's NaN Close is ever used to mark it.

    With `spreads` (a HalfSpreads), each hedging call also gets a HalfSpread field,
    its half-spread per unit on the rows it can trade, which yabte charges as a cost.
    """
    prices = option_prices(market, rolls)
    prices[SPOT] = market.spot
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
    cols = {}
    for name in prices.columns:
        cols[(name, "Close")] = prices[name]
        cols[(name, "Open")] = prices[name].shift(1)
    if spreads is not None:
        spot = prices[SPOT]
        for name, opt in hedging_calls(rolls).items():
            hs = half_spread_column(spreads, opt, spot)
            cols[(name, "HalfSpread")] = hs.where(cols[(name, "Open")].notna())
    return pd.DataFrame(cols).sort_index(axis=1)


def make_assets(data, spreads=None):
    """yabte assets for build_data's frame, filling at Open. With `spreads`, the
    underlying pays spreads.stock_cost_bps and the hedging calls their HalfSpread
    field; the straddle legs trade free."""
    assets = []
    for name in data.columns.get_level_values(0).unique():
        cost = {}
        if spreads is not None and name == SPOT:
            cost = dict(cost=str(spreads.stock_cost_bps), cost_type=CostType.BPS)
        elif spreads is not None and (name, "HalfSpread") in data.columns:
            cost = dict(cost_field="HalfSpread", cost_field_type=CostType.PER_UNIT)
        assets.append(
            OHLCAsset(
                name=name,
                fill_price="open",
                price_round_dp=6,
                quantity_round_dp=6,
                **cost,
            )
        )
    return assets


def metrics_frame(srr, method):
    """One method's per-rebalance metrics (HedgeStrategy's record()), indexed by
    the close traded at."""
    strat = next(s for s in srr.strategies if s.method == method)
    return strat.recorded.set_index("date").rename_axis(None).sort_index()


def book_value(srr, method, net=True):
    """A book's daily total, net of the transaction costs yabte charged; with
    net=False, before costs (adding them back is exact, since costs only move cash
    and every order is a target position)."""
    h = srr.book_history[method]
    total = h.total.astype(float)
    return total if net else total + h.costs.astype(float)


def hedge_costs(srr, method):
    """Transaction costs a method's book paid on each row."""
    paid = srr.book_history[method].costs.astype(float)
    return paid.diff().fillna(paid.iloc[0])


def tracking_error(srr, rolls, method, net=True):
    """
    Z_t within each roll: book value minus its value at the roll's entry close, net
    of transaction costs unless net=False.
    """
    total = book_value(srr, method, net)
    z = pd.Series(np.nan, index=total.index)
    for roll in rolls:
        live = (total.index > roll.entry) & (total.index <= roll.expiry)
        z[live] = total[live] - total.loc[roll.entry]
    return z.dropna()


def summarise(srr, rolls, methods=METHODS, net=True):
    """
    Z statistics pool Z_t over every live day of every roll, as in the paper's
    Table 11, so they mix horizons (day 1 to expiry). The daily dZ and terminal Z_T
    rows give the one-day and whole-roll views separately. Every row is net of the
    transaction costs charged in the backtest unless net=False; "hedge cost / roll"
    is added when net and any costs were charged.
    """
    paid = {m: hedge_costs(srr, m).sum() for m in methods}
    rows = {}
    for method in methods:
        z = tracking_error(srr, rolls, method, net)
        total = book_value(srr, method, net)
        dz = pd.concat(  # one-day changes within each roll only
            [
                total[(total.index >= r.entry) & (total.index <= r.expiry)].diff()
                for r in rolls
            ]
        ).dropna()
        terminal = np.array([z.loc[r.expiry] for r in rolls if r.expiry in z.index])
        rows[method] = {
            "Z mean": z.mean(),
            "Z std": z.std(),
            "5% VaR": -z.quantile(0.05),
            "2.5% VaR": -z.quantile(0.025),
            "1% VaR": -z.quantile(0.01),
            "daily dZ std": dz.std(),
            "terminal Z_T std": terminal.std(ddof=1),
            "terminal |Z_T| mean": np.abs(terminal).mean(),
        }
        if net and any(paid.values()):
            rows[method]["hedge cost / roll"] = paid[method] / len(rolls)
    return pd.DataFrame(rows)


def run_backtest(
    results,
    market,
    start=None,
    end=None,
    strike_mult=1.2,
    hold_days=21,
    quantity=100.0,
    n_scenarios=1000,
    seed=0,
    device="cpu",
    methods=METHODS,
    engine="python",
    hedge_strike="current",
    max_hedge_ratio=None,
    restrike_band=0.02,
    spreads=None,
):
    """
    Run every method in `methods` on back-to-back straddles; returns (srr, rolls,
    scenarios). start/end default to the model's test period (see main's --help).
    `engine` is yabte's: "python" or "cpp" give the same results. hedge_strike,
    restrike_band and max_hedge_ratio: see make_rolls and hedge_ratios. `spreads`
    (a transaction_costs.HalfSpreads) makes yabte charge hedge trades as they
    trade; None is frictionless.
    """
    # Out of sample: hedge from the close before the first test date, so every
    # scenario predicts a test-set day.
    n_test = results.condition_test.shape[0]
    test_dates = pd.DatetimeIndex(results.dates_t[-n_test:])
    first_oos = market.dates[market.dates.get_loc(test_dates[0]) - 1]
    start_d = first_oos
    if start:
        start_d = market.dates[market.dates >= pd.Timestamp(start)][0]
    # Default end: the test period's end, or the last market date if start is later.
    end_d = test_dates[-1] if start_d <= test_dates[-1] else market.dates[-1]
    if end:
        end_d = market.dates[market.dates <= pd.Timestamp(end)][-1]
    end_d = min(end_d, market.dates[-2])  # the last hedge needs a next day
    if start_d < first_oos:
        warnings.warn(
            f"Hedging from {start_d.date()} overlaps the model's training data, which "
            f"ends before {test_dates[0].date()}: results are in sample."
        )

    if np.isclose(strike_mult, 1.0):
        warnings.warn(
            "strike_mult 1.0: bs_delta_vega and volgan_atm replicate the straddle "
            "exactly via put-call parity, so their results are not informative."
        )
    rolls = make_rolls(
        market, start_d, end_d, hold_days, strike_mult, hedge_strike, restrike_band
    )
    if not rolls:
        raise SystemExit(
            f"No complete {hold_days}-day roll fits between {start_d.date()} and "
            f"{end_d.date()}; reduce --hold-days."
        )
    end_d = rolls[-1].expiry
    data = build_data(market, rolls, start_d, end_d, spreads)
    # Fill at Open, which build_data sets to the previous close.
    assets = make_assets(data, spreads)

    scenarios = VolGANScenarios(results, n_scenarios, seed, device)
    runner = StrategyRunner(
        data=data,
        assets=assets,
        strategies=[HedgeStrategy(method=m, book=m) for m in methods],
        books=[Book(name=m, cash="0") for m in methods],
        engine=engine,
    )
    srr = runner.run(
        {
            "market": market,
            "rolls": rolls,
            "quantity": quantity,
            "scenarios": scenarios,
            "max_hedge_ratio": max_hedge_ratio,
        }
    )
    return srr, rolls, scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model", type=Path, default=root_dir / "data-out/volgan_res.pkl"
    )
    parser.add_argument(
        "--price-data-path", type=Path, default=root_dir / "data-in/AAPL-prices.csv"
    )
    parser.add_argument(
        "--surface-path",
        type=Path,
        default=root_dir / "data-out/surfaces_transform_enhanced.csv",
    )
    parser.add_argument(
        "--strike-mult",
        type=float,
        default=1.2,
        help="K / S0. At 1.0 the straddle's call is the ATM hedging call, so the "
        "delta-vega and VolGAN+ATM hedges replicate it exactly by put-call parity.",
    )
    parser.add_argument("--hold-days", type=int, default=21, help="trading days")
    parser.add_argument("--quantity", type=float, default=100.0, help="straddles")
    parser.add_argument("--n-scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="scenario seed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--hedge-strike",
        choices=["current", "entry"],
        default="current",
        help="strike of the delta-vega / VolGAN+ATM hedging call: re-struck at the "
        "money (current), or fixed at the entry spot (entry; its vega vanishes as "
        "spot moves away, giving huge hedge positions)",
    )
    parser.add_argument(
        "--restrike-band",
        type=float,
        default=0.02,
        help="with --hedge-strike current, re-strike when spot has moved more than "
        "this fraction from the call's strike (0: every day)",
    )
    parser.add_argument(
        "--cost-mult",
        type=float,
        default=1.0,
        help="scale on the measured bid/ask half-spreads; 0 is frictionless",
    )
    parser.add_argument(
        "--stock-cost-bps",
        type=float,
        default=1.0,
        help="half-spread plus impact for the underlying, basis points of spot",
    )
    parser.add_argument("--spread-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument(
        "--max-hedge-ratio",
        type=float,
        help="calls per straddle; above it the vega methods hedge delta only that day. "
        "Default: no limit",
    )
    parser.add_argument(
        "--engine",
        choices=["python", "cpp"],
        default="python",
        help="yabte engine; both give the same results",
    )
    parser.add_argument(
        "--start",
        help="first hedging date (YYYY-MM-DD). Default: the close before the model's "
        "first test date. Later periods need a --surface-path that covers them.",
    )
    parser.add_argument(
        "--end",
        help="last date (YYYY-MM-DD). Default: the last test date, or the last date "
        "in --surface-path if --start is after the test period",
    )
    parser.add_argument("--output", type=Path, help="CSV of daily Z_t per method")
    args = parser.parse_args()

    results = load_results(args.model, map_location=args.device)
    market = Market(args.price_data_path, args.surface_path)
    spreads = HalfSpreads.from_csv(
        args.spread_table, stock_bps=args.stock_cost_bps, multiplier=args.cost_mult
    )
    srr, rolls, _ = run_backtest(
        results,
        market,
        start=args.start,
        end=args.end,
        strike_mult=args.strike_mult,
        hold_days=args.hold_days,
        quantity=args.quantity,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        device=args.device,
        engine=args.engine,
        hedge_strike=args.hedge_strike,
        max_hedge_ratio=args.max_hedge_ratio,
        restrike_band=args.restrike_band,
        spreads=spreads,
    )
    end = rolls[-1].expiry

    print(
        f"{len(rolls)} straddles x {args.quantity:g}, K = {args.strike_mult:g} x S0, "
        f"{rolls[0].entry.date()} to {end.date()}, "
        f"{args.n_scenarios} VolGAN scenarios/day ({', '.join(VOLGAN_METHODS)}); "
        f"costs {args.cost_mult:g} x measured half-spreads, stock "
        f"{args.stock_cost_bps:g} bp"
    )
    with pd.option_context(
        "display.float_format",
        "{:,.2f}".format,
        "display.width",
        120,
        "display.max_columns",
        None,
    ):
        print(summarise(srr, rolls))

    if args.output:
        pd.DataFrame({m: tracking_error(srr, rolls, m) for m in METHODS}).to_csv(
            args.output
        )
        print(f"Saved daily tracking errors to {args.output}")
    return srr, rolls


if __name__ == "__main__":
    main()
