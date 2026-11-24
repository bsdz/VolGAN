"""
Option hedging with VolGAN scenarios, following Vuletić & Cont (2025), Section 5.

Options are priced with Black-Scholes (r = 0, matching smallBS in VolGAN.py) at the
implied vol read off a (real or simulated) VolGAN surface. Hedge ratios come from:

- "bs_delta": Black-Scholes delta, hedging with the underlying (eq. 25).
- "bs_delta_vega": delta-vega neutral, hedging with the underlying and an at-the-money
  call with the straddle's expiry (eq. 26).
- "volgan_delta": regression of the portfolio's one-day change on the underlying's
  change across VolGAN next-day scenarios (eq. 27, underlying only).
- "volgan_atm": the same regression with the underlying and the ATM call (eq. 27).
- "unhedged": no hedge, as a reference.

By default the hedging call is re-struck at the money whenever spot has moved more
than 2% from its strike (make_rolls' hedge_strike="current", restrike_band=0.02). With hedge_strike="entry" it is struck at the entry spot and
held to expiry; once spot moves away it has almost no vega, and the vega hedges
then hold huge offsetting positions (up to ~1e8 calls per 100 straddles).

This module has no dependency on the backtesting engine; see hedging_backtest.py.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from VolGAN import LK, LT, M_GRID, TAU_GRID, DataPreprocesssing, VolGANData

METHODS = ("unhedged", "bs_delta", "bs_delta_vega", "volgan_delta", "volgan_atm")
VOLGAN_METHODS = ("volgan_delta", "volgan_atm")


# ---------------------------------------------------------------------------
# Pricing off a surface
# ---------------------------------------------------------------------------


def interp_iv(surfaces, m, tau):
    """
    Bilinear interpolation of implied vol on the (M_GRID, TAU_GRID) grid.

    surfaces: (..., LK * LT) flattened time-major vols (not log vols).
    m, tau: moneyness K/S and time to expiry in years, broadcastable to
        surfaces.shape[:-1]. Values off the grid are clamped (flat extrapolation),
        so maturities under 30 days use the 30-day vol.
    """
    surfaces = np.asarray(surfaces, dtype=float)
    shape = surfaces.shape[:-1]
    m = np.broadcast_to(np.clip(m, M_GRID[0], M_GRID[-1]), shape)
    tau = np.broadcast_to(np.clip(tau, TAU_GRID[0], TAU_GRID[-1]), shape)

    i = np.clip(np.searchsorted(M_GRID, m) - 1, 0, LK - 2)
    j = np.clip(np.searchsorted(TAU_GRID, tau) - 1, 0, LT - 2)
    wm = (m - M_GRID[i]) / (M_GRID[i + 1] - M_GRID[i])
    wt = (tau - TAU_GRID[j]) / (TAU_GRID[j + 1] - TAU_GRID[j])

    def at(ii, jj):  # vol at grid node (ii, jj); flat index is jj * LK + ii
        idx = (jj * LK + ii)[..., None]
        return np.take_along_axis(surfaces, idx, axis=-1)[..., 0]

    return (
        (1 - wm) * (1 - wt) * at(i, j)
        + wm * (1 - wt) * at(i + 1, j)
        + (1 - wm) * wt * at(i, j + 1)
        + wm * wt * at(i + 1, j + 1)
    )


def black_scholes(S, K, tau, sigma, call):
    """Price, delta and vega with r = q = 0. At or after expiry: intrinsic value."""
    S, K, tau, sigma = np.broadcast_arrays(*map(np.asarray, (S, K, tau, sigma)))
    live = tau > 0
    t = np.where(live, tau, 1.0)
    sqt = sigma * np.sqrt(t)
    d1 = (np.log(S / K) + 0.5 * sqt**2) / sqt
    d2 = d1 - sqt
    if call:
        price = S * norm.cdf(d1) - K * norm.cdf(d2)
        delta = norm.cdf(d1)
        intrinsic, idelta = np.maximum(S - K, 0.0), (S > K).astype(float)
    else:
        price = K * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        intrinsic, idelta = np.maximum(K - S, 0.0), -(S < K).astype(float)
    vega = S * norm.pdf(d1) * np.sqrt(t)
    return (
        np.where(live, price, intrinsic),
        np.where(live, delta, idelta),
        np.where(live, vega, 0.0),
    )


@dataclass(frozen=True)
class Option:
    name: str
    strike: float
    expiry: pd.Timestamp
    call: bool

    def tau(self, date):
        return (self.expiry - pd.Timestamp(date)).days / 365.0

    def vol(self, date, S, surfaces):
        """Implied vol at this option's moneyness and maturity on `date`."""
        S = np.asarray(S, dtype=float)
        return interp_iv(surfaces, self.strike / S, max(self.tau(date), 0.0))

    def greeks(self, date, S, surfaces):
        """Price, delta and vega (per unit vol) on `date` for spot(s) S and surface(s)."""
        sigma = self.vol(date, S, surfaces)
        return black_scholes(S, self.strike, self.tau(date), sigma, self.call)

    def price(self, date, S, surfaces):
        return self.greeks(date, S, surfaces)[0]


@dataclass(frozen=True)
class Roll:
    """One straddle position: opened at `entry`'s close, held to `expiry`."""

    entry: pd.Timestamp
    expiry: pd.Timestamp
    call: Option
    put: Option
    atm_call: Option  # the hedging call at entry, struck at the entry spot
    # Re-struck hedging calls by date (hedge_strike="current"); empty means atm_call
    # is the hedge throughout.
    hedges: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def legs(self):
        return (self.call, self.put)

    def hedge_call(self, date):
        """The call the delta-vega and VolGAN+ATM methods hedge with at `date`."""
        return self.hedges.get(pd.Timestamp(date), self.atm_call)


# ---------------------------------------------------------------------------
# Market data and scenarios
# ---------------------------------------------------------------------------


class Market:
    """Daily closes, real surfaces and VolGAN condition vectors, indexed by date."""

    def __init__(self, datapath, surfacepath):
        surfaces, prices, _, _, _, _, _, _, dates = VolGANData(datapath, surfacepath)
        self.dates = pd.DatetimeIndex(dates)
        self.spot = pd.Series(prices, index=self.dates)
        self.surfaces = pd.DataFrame(surfaces, index=self.dates)
        _, condition, *_, dates_t, _, _, _, _ = DataPreprocesssing(
            datapath, surfacepath
        )
        # condition row for date d uses information up to the previous trading day only
        self.condition = pd.DataFrame(condition, index=pd.DatetimeIndex(dates_t))

    def next_date(self, date):
        i = self.dates.get_loc(pd.Timestamp(date))
        return self.dates[i + 1]


class VolGANScenarios:
    """Next-day (spot, surface) scenarios from a trained VolGAN generator."""

    def __init__(self, results, n_scenarios=1000, seed=0, device="cpu"):
        self.gen = results.gen.to(device).eval()
        self.noise_dim = results.noise_dim
        self.n = n_scenarios
        self.seed = seed
        self.device = device
        self._cache = {}

    def next_day(self, market, date):
        """Returns (next_date, spots (n,), surfaces (n, LK*LT)) for the day after date."""
        date = pd.Timestamp(date)
        if date not in self._cache:
            nxt = market.next_date(date)
            cond = torch.tensor(
                market.condition.loc[nxt].to_numpy(),
                dtype=torch.float,
                device=self.device,
            ).repeat(self.n, 1)
            # seeded per date, so every method sees the same scenarios for a date
            g = torch.Generator(device=self.device).manual_seed(
                self.seed * 100_003 + int(date.strftime("%Y%m%d"))
            )
            noise = torch.randn(
                (self.n, self.noise_dim), generator=g, device=self.device
            )
            with torch.no_grad():
                out = self.gen(noise, cond).cpu().numpy()
            log_ret = out[:, 0] / np.sqrt(
                252
            )  # generator returns annualised log-returns
            spots = market.spot.loc[date] * np.exp(log_ret)
            surfaces = np.exp(out[:, 1:] + cond[:, 3:].cpu().numpy())
            self._cache[date] = (nxt, spots, surfaces)
        return self._cache[date]


# ---------------------------------------------------------------------------
# Hedge ratios
# ---------------------------------------------------------------------------


def regression_hedge(d_value, d_instruments):
    """OLS of d_value (n,) on d_instruments (n, k) with an intercept; returns (k,)."""
    X = np.column_stack([np.ones(len(d_value)), d_instruments])
    coef, *_ = np.linalg.lstsq(X, d_value, rcond=None)
    return coef[1:]


def hedge_ratios(
    method, market, date, roll, quantity=1.0, scenarios=None, max_hedge_ratio=None
):
    """
    Hedge ratios at `date`'s close for a long position of `quantity` straddles.

    Returns {"spot": phi_0, "hedge": phi_1}, phi_1 in units of roll.hedge_call(date):
    the positions that replicate the straddle (a hedger holding the straddle is
    short these). If max_hedge_ratio is set and |phi_1| would exceed
    max_hedge_ratio calls per straddle, the vega methods hedge delta only that day
    (bs_delta_vega as bs_delta, volgan_atm as volgan_delta).
    """
    S = market.spot.loc[date]
    surf = market.surfaces.loc[date].to_numpy()
    hedge = roll.hedge_call(date)
    out = {"spot": 0.0, "hedge": 0.0}

    def too_big(phi_1):
        return max_hedge_ratio is not None and abs(phi_1) > max_hedge_ratio * quantity

    if method == "unhedged":
        return out

    greeks = [leg.greeks(date, S, surf) for leg in roll.legs]
    if method in ("bs_delta", "bs_delta_vega"):
        delta_v = quantity * sum(g[1] for g in greeks)
        if method == "bs_delta":
            out["spot"] = float(delta_v)
            return out
        vega_v = quantity * sum(g[2] for g in greeks)
        _, delta_h, vega_h = hedge.greeks(date, S, surf)
        phi_1 = vega_v / vega_h if vega_h > 1e-12 else 0.0
        if too_big(phi_1):
            phi_1 = 0.0
        out["spot"] = float(delta_v - phi_1 * delta_h)
        out["hedge"] = float(phi_1)
        return out

    if method in VOLGAN_METHODS:
        nxt, spots, surfaces = scenarios.next_day(market, date)
        value_now = quantity * sum(g[0] for g in greeks)
        value_next = quantity * sum(
            leg.price(nxt, spots, surfaces) for leg in roll.legs
        )
        d_spot = spots - S
        d_value = value_next - value_now
        # The day before expiry, the hedging call is worth its intrinsic value
        # tomorrow: it has no vol exposure and its change is collinear with d_spot
        # (or constant), so the two-instrument regression is rank deficient. Hedge
        # with the underlying alone that day.
        if method == "volgan_atm" and hedge.tau(nxt) > 0:
            d_atm = hedge.price(nxt, spots, surfaces) - hedge.price(date, S, surf)
            phi_0, phi_1 = regression_hedge(d_value, np.column_stack([d_spot, d_atm]))
            if not too_big(phi_1):
                out["spot"], out["hedge"] = float(phi_0), float(phi_1)
                return out
        (phi_0,) = regression_hedge(d_value, d_spot[:, None])
        out["spot"] = float(phi_0)
        return out

    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def position_metrics(market, date, roll, quantity, phi):
    """
    Vols, Black-Scholes greeks and hedge ratios at `date`'s close for a long
    position of `quantity` straddles hedged with `phi` (from hedge_ratios). The
    "atm" fields are for that day's hedging call, roll.hedge_call(date).

    Vegas are per vol point (0.01). net_delta / net_vega are the Black-Scholes
    exposures left after the hedge (straddles minus the replicating positions), so
    they are zero for bs_delta_vega and show what the other methods leave open.
    """
    S = float(market.spot.loc[date])
    surf = market.surfaces.loc[date].to_numpy()
    hedge = roll.hedge_call(date)
    out = {"roll": roll.entry, "spot": S, "days_to_expiry": (roll.expiry - date).days}
    out["strike_atm"] = hedge.strike
    for key, opt in (("call", roll.call), ("put", roll.put), ("atm", hedge)):
        price, delta, vega = (float(x) for x in opt.greeks(date, S, surf))
        out[f"iv_{key}"] = float(opt.vol(date, S, surf))
        out[f"price_{key}"] = price
        out[f"delta_{key}"] = delta
        out[f"vega_{key}"] = vega / 100
    for g in ("price", "delta", "vega"):
        out[f"{g}_straddle"] = quantity * (out[f"{g}_call"] + out[f"{g}_put"])
    phi_0, phi_1 = phi["spot"], phi["hedge"]
    out["phi_spot"], out["phi_atm"] = phi_0, phi_1
    out["net_delta"] = out["delta_straddle"] - phi_0 - phi_1 * out["delta_atm"]
    out["net_vega"] = out["vega_straddle"] - phi_1 * out["vega_atm"]
    return out


def make_rolls(
    market,
    start,
    end,
    hold_days=21,
    strike_mult=1.0,
    hedge_strike="current",
    restrike_band=0.02,
):
    """
    Back-to-back straddles from `start` to `end`: each opens at a close, expires
    `hold_days` trading days later, and the next one opens at that expiry's close.

    hedge_strike: "current" re-strikes the hedging call at the money (same expiry)
    at each close where spot has moved more than `restrike_band` (relative) from
    its strike, so it stays near the money without paying the spread every day
    (0 re-strikes at every close). "entry" keeps the call
    struck at the entry spot for the whole roll.
    """
    if hedge_strike not in ("current", "entry"):
        raise ValueError(
            f"hedge_strike must be 'current' or 'entry', not {hedge_strike!r}"
        )
    dates = market.dates[(market.dates >= start) & (market.dates <= end)]
    rolls, k = [], 0
    while k + hold_days < len(dates):
        entry, expiry = dates[k], dates[k + hold_days]
        S0 = float(market.spot.loc[entry])
        tag = entry.strftime("%Y%m%d")
        K = strike_mult * S0
        atm_call = Option(f"ATM{tag}", S0, expiry, True)
        hedges = {}
        if hedge_strike == "current":
            hedge = atm_call  # entry's hedge
            for d in dates[k + 1 : k + hold_days]:
                S = float(market.spot.loc[d])
                if abs(S / hedge.strike - 1) > restrike_band:
                    hedge = Option(f"ATM{tag}_{d:%Y%m%d}", S, expiry, True)
                hedges[d] = hedge
        rolls.append(
            Roll(
                entry=entry,
                expiry=expiry,
                call=Option(f"C{tag}", K, expiry, True),
                put=Option(f"P{tag}", K, expiry, False),
                atm_call=atm_call,
                hedges=hedges,
            )
        )
        k += hold_days
    return rolls


def option_prices(market, rolls):
    """
    Daily marks of every roll's options from the real surfaces (NaN when not held).

    A re-struck hedging call is marked only while it is held: from the close it is
    bought at to the close it is sold at.
    """
    cols = {}

    def mark(opt, dates):
        S = market.spot.loc[dates].to_numpy()
        surf = market.surfaces.loc[dates].to_numpy()
        cols[opt.name] = pd.Series(
            [float(opt.price(d, s, v)) for d, s, v in zip(dates, S, surf)],
            index=dates,
        )

    for roll in rolls:
        live = market.dates[
            (market.dates >= roll.entry) & (market.dates <= roll.expiry)
        ]
        for opt in (*roll.legs, roll.atm_call):
            mark(opt, live)
        held = {}
        for d, opt in roll.hedges.items():
            if opt is not roll.atm_call:
                held.setdefault(opt, []).append(d)
        for opt, days in held.items():
            mark(
                opt,
                market.dates[
                    (market.dates >= days[0])
                    & (market.dates <= market.next_date(days[-1]))
                ],
            )
    return pd.DataFrame(cols).reindex(market.dates)
