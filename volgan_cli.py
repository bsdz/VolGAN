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
from letsberational import ImpliedBlackVolatility
from tqdm import tqdm

from datacleaning import Smooth
from VolGAN import (
    VolGAN,
    VolGANData,
    VolGANResults,
    arbitrage_penalty_tensor,
    detangle_kt,
    entangle_kt,
    penalty_mutau_tensor,
    smallBS_tensor,
)

root_dir = Path(__file__).parent
data_in = root_dir / "data-in"
data_out = root_dir / "data-out"


def calculate_implied_volatility(S, K, T, r, market_price, option_type="call"):
    """
    Calculate implied volatility using LetsBeRational if available.
    """
    # ImpliedBlackVolatility(price, S, K, T, q)
    # q = 1 for call, -1 for put
    q = 1 if option_type == "call" else -1

    try:
        iv = ImpliedBlackVolatility(market_price, S, K, T, q)
        return iv
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

    # Iterate over the dataframe rows to build the cloud of points
    for _, row in daily_df.iterrows():
        K = float(row["strike"])
        dte_days = float(row["dte"])

        # Skip invalid DTE
        if dte_days <= 0:
            continue

        tau = dte_days / 365.0
        m = K / underlying_price  # Moneyness definition for VolGAN (Strike / Spot)

        # IV Calculation
        # Prefer pre-calculated if available/trusted, but instructions implied using tools.
        # Let's check if 'c_iv' exists and is valid, otherwise calc.
        iv = None
        if "c_iv" in row and pd.notna(row["c_iv"]):
            try:
                val = float(row["c_iv"])
                if 0.0 < val < 5.0:
                    iv = val
            except:
                pass

        if iv is None:
            price = row["c_last"]
            try:
                price = float(price)
                if price > 0:
                    iv = calculate_implied_volatility(
                        underlying_price, K, tau, r, price, "call"
                    )
            except:
                pass

        if iv is not None and 0.0 < iv < 5.0:
            m_list.append(m)
            tau_list.append(tau)
            iv_list.append(iv)

    return np.array(m_list), np.array(tau_list), np.array(iv_list)


def generate_surface(input_price_file, input_vol_file, output_surface_file):
    """
    Orchestrator function: Load Options -> Generate Surface Grid -> Save
    """
    print(f"Loading options data from {input_vol_file}...")

    if not input_vol_file.exists():
        print(f"Error: File {input_vol_file} not found.")
        return

    # Read CSV
    try:
        df = pd.read_csv(input_vol_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Clean columns: remove brackets, lower case, strip
    df.columns = [
        c.replace("[", "").replace("]", "").strip().lower() for c in df.columns
    ]

    # Ensure necessary columns exist
    required_cols = {"quote_date", "underlying_last", "dte", "strike", "c_last"}
    if not required_cols.issubset(df.columns):
        print(f"Error: Missing columns. Found: {df.columns}")
        print(f"Required: {required_cols}")
        return

    # Convert date
    print("Processing dates...")
    df["date_dt"] = pd.to_datetime(df["quote_date"])

    # Sort by date
    df = df.sort_values("date_dt")

    unique_dates = df["date_dt"].unique()
    print(f"Found {len(unique_dates)} unique dates.")

    # Define Target Grid
    # 10 moneyness points x 8 time points
    lk = 10
    lt = 8
    m_grid = np.linspace(0.6, 1.4, lk)
    tau_grid_days = np.array([30, 60, 91, 122, 152, 182, 273, 365])
    tau_grid = tau_grid_days / 365.0

    surfaces_data = []

    # Kernel bandwidths for smoothing
    # These may need tuning.
    # h1 (moneyness) ~ 0.05-0.1, h2 (time) ~ 0.1-0.2
    h1 = 0.05
    h2 = 0.1

    # Process each day
    for date_val in tqdm(unique_dates, desc="Generating Surfaces"):
        daily_df = df[df["date_dt"] == date_val]

        if daily_df.empty:
            continue

        # Get underlying price
        S = daily_df["underlying_last"].iloc[0]

        # Prepare data for interpolation
        m_in, tau_in, iv_in = prepare_daily_data(daily_df, S)

        if len(iv_in) < 10:
            # Let's output a warning and fill with mean or 0.25.
            # print(f"Warning: Not enough data for {date_val}, using fallback.")
            surface_flat = np.full(lk * lt, 0.25)
            surfaces_data.append(surface_flat)
            continue

        # Smooth / Interpolate
        # Smooth returns matrix (len(m_want), len(tau_want)) -> (lk, lt)
        surface_matrix = Smooth(iv_in, m_in, tau_in, m_grid, tau_grid, h1, h2)

        # Handle NaNs (extrapolation issues)
        if np.isnan(surface_matrix).any():
            # Fill NaNs with nearest valid or mean
            # Simple fill with mean of valid values
            valid_mean = np.nanmean(surface_matrix)
            if np.isnan(valid_mean):
                valid_mean = 0.25
            surface_matrix = np.nan_to_num(surface_matrix, nan=float(valid_mean))

        # Flatten using entangle_kt
        surface_vector = entangle_kt(surface_matrix, lk, lt)

        surfaces_data.append(surface_vector)

    # Convert to DataFrame
    df_out = pd.DataFrame(
        surfaces_data, index=unique_dates, columns=[f"vol_{i}" for i in range(lk * lt)]
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


def train(datapath, surfacepath, device) -> VolGANResults:
    tr = 0.85
    noise_dim = 32
    hidden_dim = 16
    n_epochs = 10000
    n_grad = 25

    return VolGAN(
        datapath,
        surfacepath,
        tr,
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        n_grad=n_grad,
        lrg=0.0001,
        lrd=0.0001,
        batch_size=100,
        device=device,
    )


def simulate(
    volgan_res: VolGANResults,
    device: str,
):

    (
        gen,
        gen_opt,
        disc,
        disc_opt,
        true_train,
        true_val,
        true_test,
        condition_train,
        condition_val,
        condition_test,
        dates_t,
        m,
        tau,
        ms,
        taus,
    ) = volgan_res

    # Move models to the correct device
    gen.to(device)
    disc.to(device)
    gen.eval()

    # Move tensors to the correct device
    # First, ensure they are PyTorch tensors, then move to device
    if isinstance(true_train, np.ndarray):
        true_train = torch.tensor(true_train, dtype=torch.float32)
    # if isinstance(true_val, np.ndarray):
    #     true_val = torch.tensor(true_val, dtype=torch.float32)
    if isinstance(true_test, np.ndarray):
        true_test = torch.tensor(true_test, dtype=torch.float32)
    if isinstance(condition_train, np.ndarray):
        condition_train = torch.tensor(condition_train, dtype=torch.float32)
    # if isinstance(condition_val, np.ndarray):
    #     condition_val = torch.tensor(condition_val, dtype=torch.float32)
    if isinstance(condition_test, np.ndarray):
        condition_test = torch.tensor(condition_test, dtype=torch.float32)

    true_train = true_train.to(device)
    # true_val = true_val.to(device)
    true_test = true_test.to(device)
    condition_train = condition_train.to(device)
    # condition_val = condition_val.to(device)
    condition_test = condition_test.to(device)

    noise_dim = 32

    ntr = true_train.shape[0]
    B = 10000
    dtm = tau * 365
    # calculate the panlaty matrices
    mP_t, mP_k, mPb_K = penalty_mutau_tensor(m, dtm, device)
    m_t = torch.tensor(ms, dtype=torch.float, device=device)
    t_t = torch.tensor(taus, dtype=torch.float, device=device)
    n_test = true_test.shape[0]
    tots_test = np.zeros((n_test, B))
    Pks_t_test = mP_k.unsqueeze(0).repeat((n_test, 1, 1))
    Pkbs_t_test = mPb_K.unsqueeze(0).repeat((n_test, 1, 1))
    Ks_t_test = m_t.unsqueeze(0).repeat((n_test, 1, 1))
    ts_t_test = t_t.unsqueeze(0).repeat((n_test, 1, 1))
    ms_t_test = m_t.unsqueeze(0).repeat((n_test, 1, 1))
    fk = torch.empty(size=(B, n_test, 10, 8), device=device)
    fk_ent = np.zeros((B, n_test, 80))
    fk_inc = np.zeros((B, n_test, 80))
    ret_u = np.zeros((B, n_test))
    # tots
    tots_test = np.zeros((n_test, B))
    with torch.no_grad():
        for l in tqdm(range(B)):
            # sample noise
            noise = torch.randn((n_test, noise_dim), device=device, dtype=torch.float)
            # sample from the generator
            fake = gen(noise, condition_test[:, :])
            surface_past_test = condition_test[:, 3:]
            # simulated implied vol surfaces as vectors
            fake_surface = torch.exp(fake[:, 1:] + surface_past_test)
            # simulated surfaces as a matrices
            fk_ent[l, :, :] = fake_surface.cpu().detach().numpy()
            # simulated increments
            fk_inc[l, :, :] = fake[:, 1:].cpu().detach().numpy()
            ret_u[l, :] = fake[:, 0].cpu().detach().numpy()
            for i in range(8):
                fk[l, :, :, i] = fake_surface[:, (i * 10) : ((i + 1) * 10)]
            # calculating arbitrage penalties
            Pks_t_test = mP_k.unsqueeze(0).repeat((n_test, 1, 1))
            Pkbs_t_test = mPb_K.unsqueeze(0).repeat((n_test, 1, 1))
            BS = smallBS_tensor(
                ms_t_test[:, :, :], ts_t_test[:, :, :], fk[l, :, :, :], 0, device=device
            )

            _, _, _, tot = arbitrage_penalty_tensor(
                BS,
                mP_t.unsqueeze(0).repeat((n_test, 1, 1)),
                Pks_t_test[:, :, :],
                Pkbs_t_test[:, :, :],
            )
            tots_test[:, l] = tot.cpu().detach().numpy()

    # Check arbitrage penalties
    print(
        "Sim ",
        np.mean(tots_test),
        np.std(np.mean(tots_test, axis=1)),
        np.median(np.mean(tots_test, axis=1)),
    )

    # Plot one generated surface
    print("Generating interactive 3D surface plot...")

    # Pick the first test sample and first batch generation
    sample_idx = 0
    batch_idx = 0
    surf_vec = fk_ent[batch_idx, sample_idx, :]

    lk = 10
    lt = 8

    surf_matrix = detangle_kt(surf_vec, lk, lt)

    m_grid = np.linspace(0.6, 1.4, lk)
    tau_grid_days = np.array([30, 60, 91, 122, 152, 182, 273, 365])
    tau_grid = tau_grid_days / 365.0

    # Create plotly figure
    # z axis: Implied Volatility
    # x axis: Time to Maturity
    # y axis: Moneyness
    # surf_matrix is (10, 8) -> 10 rows (moneyness), 8 cols (time).
    fig = go.Figure(data=[go.Surface(z=surf_matrix, x=tau_grid, y=m_grid)])

    fig.update_layout(
        title="Generated Volatility Surface (Sample 0)",
        scene=dict(
            xaxis_title="Time to Maturity (Years)",
            yaxis_title="Moneyness (K/S)",
            zaxis_title="Implied Volatility",
        ),
        autosize=True,
    )

    output_html = "generated_surface.html"
    fig.write_html(output_html)
    print(f"3D interactive surface plot saved to {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VolGAN ")

    # actions
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Flag to indicate whether to analyze the inputs",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Flag to indicate whether to train the model",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Flag to indicate whether to simulate the model",
    )
    parser.add_argument(
        "--generate-surface",
        action="store_true",
        help="Flag to indicate whether to generate the surface",
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
        help="Path to the implied vol data CSV file",
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
        help="Path to save the VolGAN results pickle file",
    )

    # training parameters
    parser.add_argument(
        "--epochs", type=int, default=10000, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for training"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )

    args = parser.parse_args()

    res = None

    if args.generate_surface:
        input_file_vol_surface = args.raw_vol_data_path

        # Ensure output directory exists
        Path(args.surface_path).parent.mkdir(parents=True, exist_ok=True)

        generate_surface(
            args.price_data_path, input_file_vol_surface, args.surface_path
        )

    if args.analyze:
        analyze_inputs(args.price_data_path, args.surface_path)

    if args.train:
        res = train(args.price_data_path, args.surface_path, args.device)

        dated_output_path = args.output_path.with_name(
            f"{args.output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        )

        with dated_output_path.open("wb") as fh:
            pickle.dump(res, fh)

        with args.output_path.open("wb") as fh:
            pickle.dump(res, fh)

    if args.simulate:
        if res is None:
            with args.output_path.open("rb") as fh:
                res = pickle.load(fh)

        simulate(res, args.device)
