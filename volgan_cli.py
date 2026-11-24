"""
Partially based off original example code by vuletic@maths.ox.ac.uk.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from datacleaning import Smooth
from VolGAN import (
    DEFAULT_NOISE_DIM,
    LK,
    LT,
    M_GRID,
    TAU_DAYS,
    TAU_GRID,
    VolGAN,
    VolGANData,
    VolGANResults,
    arbitrage_penalty_tensor,
    detangle_kt,
    entangle_kt,
    load_results,
    penalty_mutau_tensor,
    smallBS_tensor,
)

root_dir = Path(__file__).parent
data_in = root_dir / "data-in"
data_out = root_dir / "data-out"

# Value used for a whole surface when a day has too few usable quotes.
FALLBACK_VOL = 0.25


def calculate_implied_volatility(S, K, T, r, market_price, option_type="call"):
    """
    Calculate Black implied volatility with Let's Be Rational (pylbr).

    Note: S is used as the forward and the price is treated as undiscounted,
    so r is currently ignored.
    """
    # Imported lazily: only surface generation needs it.
    from pylbr import implied_volatility_from_a_transformed_rational_guess

    # (price, F, K, T, q), q = 1 for call, -1 for put
    q = 1 if option_type == "call" else -1

    try:
        return implied_volatility_from_a_transformed_rational_guess(
            market_price, S, K, T, q
        )
    except Exception:
        return None


def prepare_daily_data(daily_df, underlying_price, r=0.02):
    """
    Extracts moneyness, time-to-maturity, and implied volatility arrays
    from the daily dataframe for interpolation.
    """
    m_list = []
    tau_list = []
    iv_list = []

    for _, row in daily_df.iterrows():
        K = float(row["strike"])
        dte_days = float(row["dte"])

        if dte_days <= 0:
            continue

        tau = dte_days / 365.0
        m = K / underlying_price  # Moneyness definition for VolGAN (Strike / Spot)

        # Prefer the vendor IV when present and sane, otherwise invert c_last.
        iv = None
        if "c_iv" in row and pd.notna(row["c_iv"]):
            try:
                val = float(row["c_iv"])
                if 0.0 < val < 5.0:
                    iv = val
            except (TypeError, ValueError):
                pass

        if iv is None:
            try:
                price = float(row["c_last"])
            except (TypeError, ValueError):
                price = None
            if price is not None and price > 0:
                iv = calculate_implied_volatility(
                    underlying_price, K, tau, r, price, "call"
                )

        if iv is not None and 0.0 < iv < 5.0:
            m_list.append(m)
            tau_list.append(tau)
            iv_list.append(iv)

    return np.array(m_list), np.array(tau_list), np.array(iv_list)


def generate_surface(input_vol_file, output_surface_file, h1=0.01, h2=0.02):
    """
    Load option quotes, smooth each day onto the (M_GRID, TAU_GRID) grid and
    save one flattened surface per quote date.

    Spot comes from the options file's underlying_last column.
    """
    print(f"Loading options data from {input_vol_file}...")

    if not input_vol_file.exists():
        print(f"Error: File {input_vol_file} not found.")
        return

    try:
        df = pd.read_csv(input_vol_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Clean columns: remove brackets, lower case, strip
    df.columns = [
        c.replace("[", "").replace("]", "").strip().lower() for c in df.columns
    ]

    required_cols = {"quote_date", "underlying_last", "dte", "strike", "c_last"}
    if not required_cols.issubset(df.columns):
        print(f"Error: Missing columns. Found: {df.columns}")
        print(f"Required: {required_cols}")
        return

    print("Processing dates...")
    df["date_dt"] = pd.to_datetime(df["quote_date"])
    df = df.sort_values("date_dt")

    unique_dates = df["date_dt"].unique()
    print(f"Found {len(unique_dates)} unique dates.")

    surfaces_data = []
    n_fallback = 0
    n_nan_filled = 0

    # Kernel bandwidths h1 (moneyness) and h2 (maturity, years). datacleaning.kernel
    # uses exp(-x^2 / (2 h)), as in the paper's eq. 21, so they are variances. The
    # default (0.01, 0.02), i.e. standard deviations of 0.10 and 0.14 years, was chosen
    # on 16 training-period days as the knee between leave-one-out fit error and the
    # arbitrage penalty of the smoothed surface. The old (0.05, 0.1) had about 55% more
    # LOO error; the paper's SPX choice (0.002, 0.046) had about 5x the arbitrage
    # penalty on this data.
    print(f"Kernel bandwidths h1={h1}, h2={h2}")

    for date_val, daily_df in tqdm(df.groupby("date_dt"), desc="Generating Surfaces"):
        S = daily_df["underlying_last"].iloc[0]

        m_in, tau_in, iv_in = prepare_daily_data(daily_df, S)

        if len(iv_in) < 10:
            n_fallback += 1
            surfaces_data.append(np.full(LK * LT, FALLBACK_VOL))
            continue

        # Smooth returns matrix (LK, LT)
        surface_matrix = Smooth(iv_in, m_in, tau_in, M_GRID, TAU_GRID, h1, h2)

        if np.isnan(surface_matrix).any():
            n_nan_filled += 1
            valid_mean = np.nanmean(surface_matrix)
            if np.isnan(valid_mean):
                valid_mean = FALLBACK_VOL
            surface_matrix = np.nan_to_num(surface_matrix, nan=float(valid_mean))

        surfaces_data.append(entangle_kt(surface_matrix, LK, LT))

    if n_fallback or n_nan_filled:
        print(
            f"Warning: {n_fallback} days used a flat {FALLBACK_VOL} fallback surface; "
            f"{n_nan_filled} days had NaN grid points filled with the day's mean."
        )

    df_out = pd.DataFrame(
        surfaces_data,
        index=pd.DatetimeIndex(unique_dates).strftime("%Y-%m-%d"),
        columns=[f"vol_{i}" for i in range(LK * LT)],
    )

    print(f"Saving {len(df_out)} surfaces to {output_surface_file}...")
    df_out.to_csv(output_surface_file, index=True)
    print("Done.")


def analyze_inputs(datapath, surfacepath):
    surfaces_transform, prices, prices_prev, log_rtn, m, tau, ms, taus, dates_dt = (
        VolGANData(datapath, surfacepath)
    )

    realised_vol_t = np.zeros(len(log_rtn) - 21)
    for i in range(len(realised_vol_t)):
        realised_vol_t[i] = np.sqrt(252 / 21) * np.sqrt(
            np.sum(log_rtn[i : (i + 21)] ** 2)
        )

    plt.rcParams["figure.figsize"] = [15.75, 9.385]
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rcParams.update({"font.size": 12})

    plt.plot(dates_dt[21:], realised_vol_t)
    plt.title("Realised Volatility")
    plt.xlabel("Date")
    plt.ylabel("Realised Volatility")
    plt.show()


def train(
    datapath,
    surfacepath,
    device,
    n_epochs=10000,
    batch_size=100,
    noise_dim=DEFAULT_NOISE_DIM,
    hidden_dim=16,
    n_grad=25,
) -> VolGANResults:
    return VolGAN(
        datapath,
        surfacepath,
        tr=0.85,
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        n_grad=n_grad,
        lrg=0.0001,
        lrd=0.0001,
        batch_size=batch_size,
        device=device,
    )


def _to_device(x, device):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    return x.to(device)


def simulate(
    volgan_res: VolGANResults,
    device: str,
    n_sims: int = 10000,
    output_html: Path = Path("generated_surface.html"),
):
    gen = volgan_res.gen.to(device)
    gen.eval()
    condition_test = _to_device(volgan_res.condition_test, device)
    noise_dim = volgan_res.noise_dim

    n_test = condition_test.shape[0]
    dtm = volgan_res.tau * 365

    # Penalty matrices and grids, built once and broadcast over the test set
    mP_t, mP_k, mPb_K = penalty_mutau_tensor(volgan_res.m, dtm, device)
    m_t = torch.tensor(volgan_res.ms, dtype=torch.float, device=device)
    t_t = torch.tensor(volgan_res.taus, dtype=torch.float, device=device)
    Pts_test = mP_t.unsqueeze(0).repeat((n_test, 1, 1))
    Pks_test = mP_k.unsqueeze(0).repeat((n_test, 1, 1))
    Pkbs_test = mPb_K.unsqueeze(0).repeat((n_test, 1, 1))
    ms_test = m_t.unsqueeze(0).repeat((n_test, 1, 1))
    ts_test = t_t.unsqueeze(0).repeat((n_test, 1, 1))

    surface_past_test = condition_test[:, 3:]
    tots_test = np.zeros((n_test, n_sims))
    first_surface = None

    with torch.no_grad():
        for l in tqdm(range(n_sims)):
            noise = torch.randn((n_test, noise_dim), device=device, dtype=torch.float)
            fake = gen(noise, condition_test)
            # simulated implied vol surfaces as flattened vectors
            fake_surface = torch.exp(fake[:, 1:] + surface_past_test)
            if first_surface is None:
                first_surface = fake_surface[0].cpu().numpy()

            # (n_test, LK * LT) time-major -> (n_test, LK, LT)
            fk = fake_surface.reshape(n_test, LT, LK).transpose(1, 2)

            BS = smallBS_tensor(ms_test, ts_test, fk, 0, device=device)
            _, _, _, tot = arbitrage_penalty_tensor(BS, Pts_test, Pks_test, Pkbs_test)
            tots_test[:, l] = tot.cpu().numpy()

    mean_per_day = np.mean(tots_test, axis=1)
    print(
        "Arbitrage penalty  mean:",
        np.mean(tots_test),
        " std(day means):",
        np.std(mean_per_day),
        " median(day means):",
        np.median(mean_per_day),
    )

    print("Generating interactive 3D surface plot...")
    surf_matrix = detangle_kt(first_surface, LK, LT)

    fig = go.Figure(data=[go.Surface(z=surf_matrix, x=TAU_GRID, y=M_GRID)])
    fig.update_layout(
        title="Generated Volatility Surface (Sample 0)",
        scene=dict(
            xaxis_title="Time to Maturity (Years)",
            yaxis_title="Moneyness (K/S)",
            zaxis_title="Implied Volatility",
        ),
        autosize=True,
    )

    fig.write_html(output_html)
    print(f"3D interactive surface plot saved to {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VolGAN ")

    # actions
    parser.add_argument(
        "--analyze", action="store_true", help="Plot realised vol of the inputs"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--simulate", action="store_true", help="Simulate from a trained model"
    )
    parser.add_argument(
        "--generate-surface",
        action="store_true",
        help="Build the surface CSV from raw option quotes",
    )

    # paths
    parser.add_argument(
        "--price-data-path",
        type=Path,
        default=data_in / "AAPL-prices.csv",
        help="Path to the price data CSV file",
    )
    parser.add_argument(
        "--raw-vol-data-path",
        type=Path,
        default=data_in / "aapl_2016_2020.csv",
        help="Path to the raw option quotes CSV file",
    )
    parser.add_argument(
        "--surface-path",
        type=Path,
        default=data_out / "surfaces_transform_enhanced.csv",
        help="Path to the surfaces transform CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=data_out / "volgan_res.pkl",
        help="Path to save/load the VolGAN results pickle file",
    )
    parser.add_argument(
        "--h1",
        type=float,
        default=0.01,
        help="--generate-surface kernel variance in moneyness",
    )
    parser.add_argument(
        "--h2",
        type=float,
        default=0.02,
        help="--generate-surface kernel variance in maturity (years)",
    )
    parser.add_argument(
        "--html-path",
        type=Path,
        default=Path("generated_surface.html"),
        help="Where --simulate writes the 3D surface plot",
    )

    # training / simulation parameters
    parser.add_argument(
        "--epochs", type=int, default=10000, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for training"
    )
    parser.add_argument(
        "--noise-dim",
        type=int,
        default=DEFAULT_NOISE_DIM,
        help="Generator noise dimension",
    )
    parser.add_argument(
        "--n-grad",
        type=int,
        default=25,
        help="Gradient-matching epochs used to set penalty weights",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10000,
        help="Number of simulations per test day for --simulate",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed torch/numpy RNGs. Runs are repeatable on the same device and "
        "library versions; GPU results may still differ slightly between runs.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  # also seeds all CUDA devices

    res = None

    if args.generate_surface:
        Path(args.surface_path).parent.mkdir(parents=True, exist_ok=True)
        generate_surface(args.raw_vol_data_path, args.surface_path, args.h1, args.h2)

    if args.analyze:
        analyze_inputs(args.price_data_path, args.surface_path)

    if args.train:
        res = train(
            args.price_data_path,
            args.surface_path,
            args.device,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            noise_dim=args.noise_dim,
            n_grad=args.n_grad,
        )

        dated_output_path = args.output_path.with_name(
            f"{args.output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        )

        with dated_output_path.open("wb") as fh:
            pickle.dump(res, fh)

        with args.output_path.open("wb") as fh:
            pickle.dump(res, fh)

    if args.simulate:
        if res is None:
            res = load_results(args.output_path, map_location=args.device)

        simulate(res, args.device, n_sims=args.n_sims, output_html=args.html_path)
