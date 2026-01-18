import argparse
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from aiohttp import web

from VolGAN import detangle_kt

root_dir = Path(__file__).parent
data_out = root_dir / "data-out"

# Global variables to hold model and data
MODEL_RES = None
DEVICE = "cpu"


def load_model(pkl_path: Path, device_name: str):
    global MODEL_RES, DEVICE
    print(f"Loading model from {pkl_path}...")
    DEVICE = device_name

    try:
        with pkl_path.open("rb") as fh:
            MODEL_RES_RAW = pickle.load(fh)
    except FileNotFoundError:
        print(f"Error: Could not find model file at {pkl_path}")
        raise

    # Unpack for setup
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
    ) = MODEL_RES_RAW

    # Move to device
    print(f"Moving model to {DEVICE}...")
    gen.to(DEVICE)
    gen.eval()

    # Convert conditions to tensor if needed
    if isinstance(condition_test, np.ndarray):
        condition_test = torch.tensor(condition_test, dtype=torch.float32)
    condition_test = condition_test.to(DEVICE)

    # Store essential components
    MODEL_RES = {
        "gen": gen,
        "condition_test": condition_test,
    }
    print("Model loaded successfully.")


def generate_surface_plot():
    gen = MODEL_RES["gen"]
    condition_test = MODEL_RES["condition_test"]

    # Pick random sample from test set
    n_test = condition_test.shape[0]
    idx = np.random.randint(0, n_test)

    # Prepare inputs
    cond = condition_test[idx].unsqueeze(0)  # (1, cond_dim)
    # Extract past surface (log implied vol at t-1) from condition
    # In DataPreprocesssing: condition = [log_rtn_tm1, log_rtn_tm2, realised_vol_tm1, log_iv_tm1]
    # log_iv_tm1 starts at index 3
    surface_past = cond[:, 3:]

    noise_dim = 32  # Hardcoded in VolGAN architecture usually
    noise = torch.randn((1, noise_dim), device=DEVICE, dtype=torch.float)

    with torch.no_grad():
        fake = gen(noise, cond)
        # fake[:, 0] is return, fake[:, 1:] is log iv increment
        # new_log_iv = increment + past_log_iv
        # surface = exp(new_log_iv)
        fake_surface = torch.exp(fake[:, 1:] + surface_past)

    # Detangle 1D vector back to 2D matrix
    vec = fake_surface.cpu().numpy().flatten()
    lk = 10
    lt = 8
    surf_matrix = detangle_kt(vec, lk, lt)

    # Grid definition (Hardcoded matching VolGAN-cli logic)
    m_grid = np.linspace(0.6, 1.4, lk)
    tau_grid_days = np.array([30, 60, 91, 122, 152, 182, 273, 365])
    tau_grid = tau_grid_days / 365.0

    # Create Plotly figure
    fig = go.Figure(data=[go.Surface(z=surf_matrix, x=tau_grid, y=m_grid)])
    fig.update_layout(
        title=f"Generated Volatility Surface (Sample Index: {idx})",
        scene=dict(
            xaxis_title="Time to Maturity (Years)",
            yaxis_title="Moneyness (K/S)",
            zaxis_title="Implied Volatility",
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40),
        height=800,
        template="plotly_white",
    )

    # Return just the div, not full HTML
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


async def handle(request):
    try:
        plot_html = generate_surface_plot()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>This Vol Surface Does Not Exist</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; text-align: center; background: #f4f4f9; margin: 0; padding: 20px; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                h1 {{ margin-top: 0; color: #2c3e50; }}
                .subtitle {{ color: #7f8c8d; margin-bottom: 20px; }}
                button {{
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 15px 40px;
                    font-size: 18px;
                    font-weight: 600;
                    border-radius: 30px;
                    cursor: pointer;
                    margin-top: 20px;
                    box-shadow: 0 4px 6px rgba(52, 152, 219, 0.3);
                    transition: all 0.2s ease;
                }}
                button:hover {{ background-color: #2980b9; transform: translateY(-2px); box-shadow: 0 6px 8px rgba(52, 152, 219, 0.4); }}
                button:active {{ transform: translateY(0); }}
                .footer {{ margin-top: 40px; color: #95a5a6; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>This Vol Surface Does Not Exist</h1>
                <p class="subtitle">Generating synthetic arbitrage-free implied volatility surfaces using VolGAN</p>
                <div>{plot_html}</div>
                <form action="/" method="get">
                    <button type="submit">Refresh Vol Surface</button>
                </form>
                <div class="footer">
                    Trained on APPL options data 2016-2020 obtained from Kaggle datasets. 
                </div>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type="text/html")
    except Exception as e:
        return web.Response(
            text=f"<h1>Error generating surface</h1><pre>{str(e)}</pre>",
            content_type="text/html",
            status=500,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VolGAN Web Demo")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(data_out / "volgan_res.pkl"),
        help="Path to model pickle file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)",
    )

    args = parser.parse_args()

    path = Path(args.model_path)
    load_model(path, args.device)

    app = web.Application()
    app.add_routes([web.get("/", handle)])

    print(f"Starting VolGAN TVSDNE server on http://localhost:{args.port}")
    web.run_app(app, port=args.port)
