# VolGAN (Fork)

This repository contains a fork of VolGAN, a generative adversarial network for producing implied volatility surfaces while penalizing static arbitrage violations.

The original project summary can be seen in README.orig.md.

This fork extends the original code toward a more practical local workflow:

- Added a CLI workflow in volgan_cli.py for:
  - Surface generation from raw options CSVs
  - Input analysis
  - Training
  - Simulation and 3D HTML output
- Shifted data handling toward local CSV pipelines (no required Yahoo fetch path in the main forked flow).
- Added structured return types in training API (VolGANResults) for clearer downstream use.
- Improved tensor/device handling in key numerical paths to reduce CPU/GPU mismatch issues.
- Updated smoothness penalty construction and related training internals for more explicit matrix/weight handling.
- Added packaging metadata via pyproject.toml and a uv.lock for reproducible installs.
- Prices and surfaces are aligned by date, and `--seed` makes runs repeatable.

## Instructions

### 1) Setup

Requires Python 3.13+. The recommended route is [uv](https://docs.astral.sh/uv/), which installs the exact versions in `uv.lock`:

```bash
uv sync
source .venv/bin/activate
```

Or with pip (versions are not pinned this way):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Implied vols are backed out with [pylbr](https://github.com/bsdz/py_lets_be_rational), a pure-Python port of Peter Jäckel's *Let's Be Rational*. It is installed automatically from a pinned git commit.

**GPU (optional).** The default PyPI torch wheel only supports recent NVIDIA GPUs. On older cards such as Pascal (GTX 10xx) it fails with `no kernel image is available`. In that case install the CUDA 12.6 build, which still covers them:

```bash
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
```

Use `--device cpu` if you have no usable GPU. Training is slower but works.

### 2) Prepare input data

See [data-in/README.md](data-in/README.md). You need:
- a daily price history CSV with `date` (YYYY-MM-DD) and `close` columns, e.g. `data-in/AAPL-prices.csv`
- the Kaggle AAPL option chains (2016–2020), e.g. `data-in/aapl_2016_2020.csv`

Input CSVs are not committed. The generated surface file `data-out/surfaces_transform_enhanced.csv` and a trained model `data-out/volgan_res.pkl` are, so steps 3 and 5 are optional.

### 3) Generate surfaces (optional)

Smooths each day's quotes onto a 10 × 8 grid (moneyness K/S 0.6–1.4 × maturities of 30–365 days):

```bash
python volgan_cli.py --generate-surface \
  --raw-vol-data-path data-in/aapl_2016_2020.csv \
  --surface-path data-out/surfaces_transform_enhanced.csv
```

### 4) Analyze inputs

```bash
python volgan_cli.py --analyze \
  --price-data-path data-in/AAPL-prices.csv \
  --surface-path data-out/surfaces_transform_enhanced.csv
```

Prices and surfaces are matched by date. Surface dates with no price row are dropped.

### 5) Train

```bash
python volgan_cli.py --train --device cuda --seed 0
```

Notes:
- Defaults: 10000 epochs, 25 gradient-matching epochs, batch size 100, noise dimension 32. See `--help`.
- `--seed` makes runs repeatable on the same hardware and library versions. On GPU, small run-to-run differences are still possible.
- The model is saved to `data-out/volgan_res.pkl`, plus a timestamped copy.
- The committed `data-out/volgan_res.pkl` was trained with `--seed 2` (one of seeds 0–2, chosen for its hedging results below). Results vary a lot between seeds, so compare several before drawing conclusions.

### 6) Simulate and export a surface plot

```bash
python volgan_cli.py --simulate --device cuda --seed 0
```

This prints arbitrage-penalty statistics over the test period and writes an interactive surface to `generated_surface.html`. Use `--n-sims` to change the number of scenarios per day (default 10000).

Saved models load on any device. Tensors from a model trained on CUDA are remapped when it is loaded.

### 7) Hedging backtest (optional)

Backtests the paper's Section 5 hedging on rolling one-month straddles (strike 1.2 × spot) over the model's test period, using [yabte](https://github.com/bsdz/yabte). It compares no hedge, Black–Scholes delta and delta-vega hedging, and VolGAN scenario-regression hedges (underlying only, and underlying plus an at-the-money call):

```bash
uv sync --extra hedging   # or: pip install -e ".[hedging]"
python hedging_backtest.py --model data-out/volgan_res.pkl --device cpu
```

It prints tracking-error statistics per method, net of transaction costs. See `--help` for the strike, holding period, number of scenarios, cost level and how the hedging call is struck.

For a longer out-of-sample test, use the combined 2016–2023 surfaces (`data-out/surfaces_2016_2023.csv`, built from the Kaggle 2016–2020 and 2021–2023 chains with step 3):

```bash
python hedging_backtest.py --model data-out/volgan_res.pkl --device cpu \
  --surface-path data-out/surfaces_2016_2023.csv --start 2021-01-04 --end 2023-03-31
```

**Transaction costs.** Hedge trades pay half the bid/ask spread. For options, the half-spread is the median measured from the Kaggle chains' quotes for each call/put, moneyness and days-to-expiry bucket: about 3 bp of spot at the money, and 7–17 bp in the money. The table is committed as `data-out/option_half_spreads.csv` and can be regenerated with `python transaction_costs.py data-in/aapl_2016_2020.csv data-in/aapl_2021_2023.csv`. The stock pays 1 bp. Use `--cost-mult` to scale the spreads; `--cost-mult 0` gives the paper's frictionless set-up. yabte charges the costs as the hedge trades, so each book's value is net of costs. Costs are charged on hedge trades only (the stock and the hedging call). Buying the straddle is not charged, since it is the same for every method.

**The hedging call.** The delta-vega and VolGAN + ATM call hedges use an at-the-money call with the straddle's expiry. It is re-struck at the money whenever spot moves more than 2% from its strike (`--restrike-band`, 0 means every day). Two other choices are worse:

- **Struck once at entry and held** (`--hedge-strike entry`): once spot moves away, the call has almost no vega, and the vega hedges end up holding up to ~10^8 offsetting calls. Their spread costs would run into millions.
- **Re-struck every day:** this pays the spread daily and roughly doubles the hedging cost.

The 2% band was chosen because it is better for the Black–Scholes benchmark than 5% is, not tuned for VolGAN. `--max-hedge-ratio` optionally caps the position, falling back to a delta-only hedge when the cap is exceeded. The cap is off by default.

Tracking-error std (`Z std`) of the committed model (seed 2), 100 straddles at K = 1.2 × S0, 1000 scenarios/day, net of costs:

| Period | unhedged | BS delta | BS delta-vega | VolGAN delta | VolGAN + ATM call |
|---|---|---|---|---|---|
| 2020-04 to 2020-12 (test split, 8 rolls) | 795 | 111 | 68 | 207 | 88 |
| 2021-01 to 2023-03 (out of sample, 26 rolls) | 930 | 52 | 54 | 39 | 28 |

Without costs, BS delta-vega and VolGAN + ATM call are 63 and 85 (2020), and 47 and 25 (2021–2023). The delta-only hedges barely change.

Costs per roll in 2021–2023 were BS delta-vega 25, VolGAN + ATM call 15, and under 1 for the delta-only hedges. VolGAN + ATM call buys fewer hedging calls, so it pays less. In 2021–2023, after costs, BS delta-vega has a slightly larger Z std than plain BS delta. Mean |Z_T| at expiry in 2021–2023 is closer than Z std suggests: VolGAN + ATM call 45, BS delta-vega 48, VolGAN delta 55, BS delta 61.

**The results depend on the seed.** With costs, the VolGAN + ATM call Z std for seeds 0/1/2 is 309 / 60 / 88 (2020) and 66 / 31 / 28 (2021–2023). VolGAN delta is 354 / 175 / 207 and 77 / 35 / 39. Out of sample, two of the three trained models beat Black–Scholes on spread and one does not.

**Errors at expiry are biased negative for every hedge that leaves vega open.** Without costs, mean Z_T in 2021–2023 was BS delta −49, VolGAN delta −55, VolGAN + ATM call −30 and BS delta-vega −3. Implied vol exceeded realised vol, so a long straddle loses theta. Costs push the mean further down, mostly for the two hedges that trade options: with costs it is −28 for BS delta-vega and −45 for VolGAN + ATM call.

[`notebooks/VolGAN_Hedging.ipynb`](notebooks/VolGAN_Hedging.ipynb) walks through the experiment with charts: VolGAN's joint spot–vol scenarios, hedge ratios over one roll, tracking error over both periods, and the seed comparison.

### Toy - This Vol Surface Does Not Exist Demo

Starts a web server that generates 3d interactive vol surfaces (from latent space).

```bash
python volgan_tvsdne.py --port 8080 --device cuda
```

Note that the training input data is probably not very high quality (from Kaggle).

![TVSDNE](./tvsdne-screenshot.png)