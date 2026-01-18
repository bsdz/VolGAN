"""
Original by vuletic@maths.ox.ac.uk (Code to go with the VolGAN paper).

Extra enhancements and modifications by blairuk@gmail.com:

* Replaced SPY/Yahoo data fetch flow with CSV-based + loading and date/length
    alignment logic.
* Reworked smoothness penalty construction in training to explicit time/strike
    finite-difference operators & weight vectors.
* Hardened tensor device handling (in arbitrage penalty and smallBS tensor
    paths) to avoid cross-device runtime issues.
* Changed VolGAN output from positional tuple to named VolGANResults structure.
* Black autoformatting.
"""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from tqdm import tqdm


# Calculating arbitrage penalties
def penalty_matrices(K, T):
    """
    K,T: sets of strikes and times of expiry (or times to expiry, we only care about differences)
    *both sorted in an ascending order*
    P_T: such that the positive part of CP_T is the arbitrage penalty for the maturity constraint
    (the call price is an increasing function of maturity)
    P_K: such that the positive part of P_KC is the arbitrage penalty for the strike constraing
    (the call price is a decreasing function of strike)
    PB_K: such that the positive part of PB_KC is the arbitrage penalty of the butterfly constraint
    (the call price is a convex function of strike)

    C is the len(K)xlen(T) matrix containing call option prices

    """
    # initialise
    P_T = np.zeros((len(T), len(T)))
    P_K = np.zeros((len(K), len(K)))
    PB_K = np.zeros((len(K), len(K)))
    # P_T first, the last one is zero
    for j in tqdm(np.arange(0, len(T) - 1, 1)):
        P_T[j, j] = 1 / (T[j + 1] - T[j])
        P_T[j + 1, j] = -1 / (T[j + 1] - T[j])
    # now P_K and then PB_K
    for i in tqdm(np.arange(0, len(K) - 1, 1)):
        P_K[i, i] = -1 / (K[i + 1] - K[i])
        P_K[i, i + 1] = 1 / (K[i + 1] - K[i])
    # PB_K: note that it is a scaled finite difference, but let's compute it on its own just in case
    # once we fix the grid we have to run this function only once so it doesn't matter much
    for i in tqdm(np.arange(1, len(K) - 1, 1)):
        PB_K[i, i - 1] = -2 / ((K[i] - K[i - 1]) * (K[i + 1] - K[i - 1]))
        PB_K[i, i] = 2 / ((K[i] - K[i - 1]) * (K[i + 1] - K[i]))
        PB_K[i, i + 1] = -2 / ((K[i + 1] - K[i]) * (K[i + 1] - K[i - 1]))
    return P_T, P_K, PB_K


def penalty_mutau(mu, T):
    """
    Same as penalty K, T, but with moneyness instead
    """
    P_T = np.zeros((len(T), len(T)))
    P_K = np.zeros((len(mu), len(mu)))
    PB_K = np.zeros((len(mu), len(mu)))
    # P_T first, the last one is zero
    for j in np.arange(0, len(T) - 1, 1):
        P_T[j, j] = T[j] / (T[j + 1] - T[j])
        P_T[j + 1, j] = -T[j] / (T[j + 1] - T[j])
    # now P_K and then PB_K
    for i in np.arange(0, len(mu) - 1, 1):
        P_K[i, i] = -1 / (mu[i + 1] - mu[i])
        P_K[i, i + 1] = 1 / (mu[i + 1] - mu[i])
    # PB_K: note that it is a scaled finite difference, but let's compute it on its own just in case
    # once we fix the grid we have to run this function only once so it doesn't matter much
    for i in np.arange(1, len(mu) - 1, 1):
        PB_K[i, i - 1] = -(mu[i + 1] - mu[i]) / (
            (mu[i] - mu[i - 1]) * (mu[i + 1] - mu[i])
        )
        PB_K[i, i] = (mu[i + 1] - mu[i - 1]) / (
            (mu[i] - mu[i - 1]) * (mu[i + 1] - mu[i])
        )
        PB_K[i, i + 1] = -(mu[i] - mu[i - 1]) / (
            (mu[i] - mu[i - 1]) * (mu[i + 1] - mu[i])
        )
    return P_T, P_K, PB_K


def penalty_tensor(K, T, device):
    """
    matrices for calculating the arbitrage penalty (tensors)
    """
    P_T = torch.zeros(size=(len(T), len(T)), dtype=torch.float, device=device)
    P_K = torch.zeros(size=(len(K), len(K)), dtype=torch.float, device=device)
    PB_K = torch.zeros(size=(len(K), len(K)), dtype=torch.float, device=device)
    # P_T first, the last one is zero
    for j in tqdm(np.arange(0, len(T) - 1, 1)):
        P_T[j, j] = T[j] / (T[j + 1] - T[j])
        P_T[j + 1, j] = -T[j] / (T[j + 1] - T[j])
    # now P_K and then PB_K
    for i in tqdm(np.arange(0, len(K) - 1, 1)):
        P_K[i, i] = -1 / (K[i + 1] - K[i])
        P_K[i, i + 1] = 1 / (K[i + 1] - K[i])
    # PB_K: note that it is a scaled finite difference, but let's compute it on its own just in case
    # once we fix the grid we have to run this function only once so it doesn't matter much
    for i in tqdm(np.arange(1, len(K) - 1, 1)):
        PB_K[i, i - 1] = -(K[i + 1] - K[i]) / ((K[i] - K[i - 1]) * (K[i + 1] - K[i]))
        PB_K[i, i] = (K[i + 1] - K[i - 1]) / ((K[i] - K[i - 1]) * (K[i + 1] - K[i]))
        PB_K[i, i + 1] = -(K[i] - K[i - 1]) / ((K[i] - K[i - 1]) * (K[i + 1] - K[i]))
    return P_T, P_K, PB_K


def penalty_mutau_tensor(mu, T, device):
    return penalty_tensor(mu, T, device)


def arbitrage_penalty(C, P_T, P_K, PB_K):
    """
    Given the prices of calls C for a fixed grid (K,T)
    P_T, P_K, PB_K: the matrices calculated by penalty_matrices
    Returns matrix penalties (for each point) in order
    1) penalty for violating C being increasing in T
    2) penalty for violating C being decreasing in K
    3) penalty for violating C being convex in K
    plus
    4) the sum of all penalties together (scalar)
    """
    P1 = np.maximum(0, np.matmul(C, P_T))
    P2 = np.maximum(0, np.matmul(P_K, C))
    P3 = np.maximum(0, np.matmul(PB_K, C))
    return P1, P2, P3, np.sum(P1 + P2 + P3)


def arbitrage_penalty_tensor(C, P_T, P_K, PB_K):
    """
    Given the prices of calls C for a fixed grid (K,T)
    P_T, P_K, PB_K: the matrices calculated by penalty_matrices
    Returns matrix penalties (for each point) in order
    1) penalty for violating C being increasing in T
    2) penalty for violating C being decreasing in K
    3) penalty for violating C being convex in K
    plus
    4) the sum of all penalties together (scalar)
    """
    device = C.device
    zero = torch.tensor(0.0, device=device)
    P1 = torch.max(zero, torch.matmul(C, P_T))
    P2 = torch.max(zero, torch.matmul(P_K, C))
    P3 = torch.max(zero, torch.matmul(PB_K, C))
    return (
        P1,
        P2,
        P3,
        torch.sum(P1, dim=(1, 2))
        + torch.sum(P2, dim=(1, 2))
        + torch.sum(P3, dim=(1, 2)),
    )


def Black76_OptionPrice(St, tau, F, K, sigma):
    """
    St: current asset price (at time t)
    tau: time to expiry of the option
    F: forward price
    K: strike
    sigma: volatility
    ***all of the above can be both vectors and scalars***
    returns CALL PRICE(s) using the Black-76 formula
    """
    efact = F / St
    # delta of the option
    d1 = (np.log(F / K) + tau * 0.5 * sigma * sigma) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = efact * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return price


def BS_OptionPrice(St, tau, r, K, sigma):
    """
    St: current asset price (at time t)
    tau: time to expiry of the option
    r: risk-free interest rate
    K: strike
    sigma: volatility
    ***all of the above can be both vectors and scalars***
    returns CALL PRICE(s) using the Black-Scholes formula
    """
    # delta
    d1 = (np.log(St / K) + tau * (r + 0.5 * sigma * sigma)) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = St * norm.cdf(d1) - K * norm.cdf(d2) * np.exp(-r * tau)
    return price


def BS_OptionPrice_tensor(St, tau, r, K, sigma):
    """
    St: current asset price (at time t)
    tau: time to expiry of the option
    r: risk-free interest rate
    K: strike
    sigma: volatility
    ***all of the above can be both vectors and scalars***
    returns CALL PRICE(s) using the Black-Scholes formula
    """
    # delta
    norm = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    d1 = (torch.log(St / K) + tau * (r + 0.5 * sigma * sigma)) / (
        sigma * torch.sqrt(tau)
    )
    d2 = d1 - sigma * torch.sqrt(tau)
    price = St * norm.cdf(d1) - K * norm.cdf(d2) * torch.exp(-r * tau)
    price[price <= 0] = 10 ** (-10)
    return price


def smallBS(m, tau, sigma, r):
    """
    Relative call
    """
    d1 = (-np.log(m) + tau * (r + 0.5 * sigma * sigma)) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = norm.cdf(d1) - m * norm.cdf(d2) * np.exp(-r * tau)
    return price


# Converting from K,T to m,tau
def K_T_to_mu_tau(K, T, St, t):
    return K / St, T - t


def mu_tau_to_K_T(mu, tau, St, t):
    return mu * St, tau + t


def get_points(I_known, K_known, tau_known, K_want, tau_want):
    """
    given I(K,T) returns points in I(K*,tau*) using interpolation in K and T
    K_known and T_known need to be sequences
    """
    # interpolate Ks first (fix tau known)
    xvals = K_want
    S = np.zeros(shape=(len(K_want), len(tau_known)))
    for i in range(len(tau_known)):
        yinterp = np.interp(xvals, K_known, I_known[:, i])
        S[:, i] = yinterp
    # interpolate taus now
    xvals = tau_want
    I = np.zeros(shape=(len(K_want), len(tau_want)))
    xvals = tau_want
    for i in range(len(K_want)):
        yinterp = np.interp(xvals, tau_known, S[i, :])
        I[i, :] = yinterp
    return I


def get_points2(I_known, m_known, tau_known, m_want, tau_want):
    """
    given I(K,T) returns points in I(K*,tau*) using interpolation in K and T
    K_known and T_known need to be sequences
    """
    # interpolate Ks first (fix tau known)
    xvals = m_want
    S = np.zeros(shape=(len(m_want), len(tau_known)))
    for i in range(len(tau_known)):
        yinterp = np.interp(xvals, m_known, I_known[:, i])
        S[:, i] = yinterp
    # interpolate taus now
    xvals = tau_want
    I = np.zeros(shape=(len(m_want), len(tau_want)))
    xvals = tau_want
    for i in range(len(m_want)):
        yinterp = np.interp(xvals, tau_known, S[i, :])
        I[i, :] = yinterp
    return I


def plot_surface(X, Y, Z, xlabel, ylabel, zlabel, title=" "):
    """
    Plot a surface
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def combine_vectors(x, y, dim=-1):
    """
    Function for combining two tensors
    """
    combined = torch.cat([x, y], dim=dim)
    combined = combined.to(torch.float)
    return


def entangle_kt(surface, lk, lt):
    """
    from a matrix to a vector
    """
    I = np.zeros(shape=(lk * lt))
    for i in range(lt):
        I[(i * lk) : ((i + 1) * lk)] = surface[:, i]
    return I


def detangle_kt(surface, lk, lt):
    """
    from a vector to a matrix
    """
    I = np.zeros(shape=(lk, lt))
    for i in range(lt):
        I[:, i] = surface[(i * lk) : ((i + 1) * lk)]
    return I


def detangle_kt_torch(surface, lk, lt):
    I = torch.empty(size=(lk, lt))
    for i in range(lt):
        I[:, i] = surface[(i * lk) : ((i + 1) * lk)]
    return I


def scatter_surface(X, Y, Z, xlabel, ylabel, zlabel, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def smallBS_tensor(m, tau, sigma, r, device):
    """
    relative call: tensor
    """
    norm = torch.distributions.Normal(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )
    d1 = (-torch.log(m) + tau * (r + 0.5 * sigma * sigma)) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    price = norm.cdf(d1) - m * norm.cdf(d2) * torch.exp(-r * tau)
    ####avoiding numerical errors
    price[price <= 0] = 10 ** (-10)
    return price


def RelativeCall_tensor(m, tau, sigma):
    norm = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    d1 = (-torch.log(m) + tau * 0.5 * sigma * sigma) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    price = norm.cdf(d1) - m * norm.cdf(d2)
    price[price <= 0] = 10 ** (-10)
    return price


def ConvertDelta(m, tau, sigma, r=0):
    d1 = (-np.log(m) + tau * (r + 0.5 * sigma * sigma)) / (sigma * np.sqrt(tau))
    delta = norm.cdf(d1)
    return delta


def VolGANData(datapath, surfacespath):
    # read the data
    data = pd.read_csv(datapath)
    # date_dt = pd.to_datetime(data['date'],format='%d/%m/%Y')
    # cutoff_date = pd.to_datetime('01/01/2020',format='%d/%m/%Y')
    # data = data[date_dt<cutoff_date]
    # surfaces saved separately
    surfaces_transform = pd.read_csv(surfacespath, index_col=0).values
    dates = data["date"].unique()
    # data_temp = data[data["date"] == dates[0]]
    # days = data_temp["days"].unique()
    # Hardcoded days grid matching generate_aapl_surfaces_enhanced.py
    days = np.array([30, 60, 91, 122, 152, 182, 273, 365])

    prices = np.array(data["close"])

    n_surfaces = surfaces_transform.shape[0]
    n_prices = len(prices)

    if n_prices > n_surfaces:
        # Assuming alignment at the END (latest data) or START?
        # Let's assume START matching.
        prices = prices[:n_surfaces]
        # dates = dates[:n_surfaces] # If we use dates later

        # NOTE: VolGAN usually expects 'dates' to correspond to the surfaces.
        # So we should also slice 'dates'.
        dates = data["date"].unique()[:n_surfaces]
    elif n_surfaces > n_prices:
        # Trim surfaces
        surfaces_transform = surfaces_transform[:n_prices]
        dates = data["date"].unique()[:n_prices]

    m = np.linspace(0.6, 1.4, 10)
    tau = np.copy(days) / 365
    dtm = np.copy(days)
    taus, ms = np.meshgrid(tau, m)
    # penalty matrices in m and tau
    mP_t, mP_k, mPb_K = penalty_mutau(m, dtm)

    # Handling date format for plotting/tracking (optional but good to keep working)
    # AAPL-prices.csv date format is likely YYYY-MM-DD (e.g., 2014-02-06)
    # The original code expected DD/MM/YYYY or similar and did manual parsing.
    # We can just use pd.to_datetime directly on the 'dates' array which usually handles ISO format well.
    dates_dt = pd.to_datetime(dates)

    prices_prev = np.zeros(len(prices))
    prices_prev[1:] = prices[0:-1]
    # Approximate previous price for the first element
    prices_prev[0] = prices[0]

    log_rtn = np.log(prices) - np.log(prices_prev)
    return surfaces_transform, prices, prices_prev, log_rtn, m, tau, ms, taus, dates_dt


class Generator(nn.Module):
    """
    VolGAN generator
    Generator Class
    Values:
        noise_dim: the dimension of the noise, a scalar
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
        output_dim: output dimension, a scalar
    """

    def __init__(
        self,
        noise_dim,
        cond_dim,
        hidden_dim,
        output_dim,
        mean_in=False,
        std_in=False,
        mean_out=False,
        std_out=False,
    ):
        super(Generator, self).__init__()
        self.input_dim = noise_dim + cond_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        self.mu_i = mean_in
        self.std_i = std_in
        self.mu_o = mean_out
        self.std_o = std_out

        # Add the modules

        self.linear1 = nn.Linear(
            in_features=self.input_dim, out_features=self.hidden_dim
        )
        self.linear2 = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim * 2
        )
        self.linear3 = nn.Linear(
            in_features=self.hidden_dim * 2, out_features=self.output_dim
        )
        self.activation1 = nn.Softplus()
        self.activation2 = nn.Softplus()
        self.activation3 = nn.Sigmoid()

    def forward(self, noise, condition):
        """
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        """
        # x = combine_vectors(noise.to(torch.float),condition.to(torch.float),2)
        # condition: S_t-1, sigma_t-1, r_t-1, implied vol_t-1
        # out: increment in r_t, increment in implied vol _t

        # condition = (condition - self.mu_i) / self.std_i
        out = torch.cat([noise, condition], dim=-1).to(torch.float)
        out = self.linear1(out)
        out = self.activation1(out)
        out = self.linear2(out)
        out = self.activation2(out)
        out = self.linear3(out)
        # uncomment to normalise
        # out = self.mu_o + self.std_o * out
        # out = torch.max(out,torch.tensor(10**(-5)))

        return out


class Discriminator(nn.Module):
    """
    VolGAN discriminator
      in_dim: the input dimension (concatenated with the condition), a scalar
      hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, in_dim, hidden_dim, mean=False, std=False):
        super(Discriminator, self).__init__()
        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(
            in_features=self.input_dim, out_features=self.hidden_dim
        )
        self.linear2 = nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.Softplus = nn.Softplus()
        self.mu_i = mean
        self.std_i = std

    def forward(self, in_chan):
        """
        in_chan: concatenated condition with real or fake
        h_0 and c_0: for the LSTM
        """
        x = in_chan
        # uncomment to normalise
        # x = (x - self.mu_i) / self.std_i
        out = self.linear1(x)
        out = self.Softplus(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


def DataPreprocesssing(datapath, surfacepath):
    """
    function for preparing the data for VolGAN
    later to be split into train, val, test
    """
    surfaces_transform, prices, prices_prev, log_rtn, m, tau, ms, taus, dates_dt = (
        VolGANData(datapath, surfacepath)
    )
    # realised volatility at time t-1
    realised_vol_tm1 = np.zeros(len(log_rtn) - 22)
    for i in range(len(realised_vol_tm1)):
        realised_vol_tm1[i] = np.sqrt(252 / 21) * np.sqrt(
            np.sum(log_rtn[i : (i + 21)] ** 2)
        )
    # shift the time
    dates_t = dates_dt[22:]
    # log-return at t, t-1, t-2
    log_rtn_t = log_rtn[22:]
    log_rtn_tm1 = np.sqrt(252) * log_rtn[21:-1]
    log_rtn_tm2 = np.sqrt(252) * log_rtn[20:-2]
    # log implied vol at t and t-1
    log_iv_t = np.log(surfaces_transform[22:])
    log_iv_tm1 = np.log(surfaces_transform[21:-1])
    # we want to simulate the increment at time t (t - t-1)
    log_iv_inc_t = log_iv_t - log_iv_tm1

    # calculate normalisation parameters in case it is needed

    # log-returns of the underlying
    m_ret = np.mean(log_rtn_t[0:100])
    sigma_ret = np.std(log_rtn[0:100])

    # realised vol
    m_rv = np.mean(realised_vol_tm1[0:100])
    sigma_rv = np.std(realised_vol_tm1[0:100])

    # log implied vol
    m_liv = np.mean(log_iv_t[0:100, :], axis=0)
    sigma_liv = np.mean(log_iv_t[0:100, :], axis=0)

    # log implied vol increment
    m_liv_inc = np.mean(log_iv_inc_t[0:100, :], axis=0)
    sigma_liv_inc = np.mean(log_iv_inc_t[0:100, :], axis=0)

    m_in = np.concatenate(([m_ret], [m_ret], [m_rv], m_liv))
    sigma_in = np.concatenate(([sigma_ret], [sigma_ret], [sigma_rv], sigma_liv))

    # the output of the generator is the return of SPX and ret of log-iv
    m_out = np.concatenate(([m_ret], m_liv_inc))
    sigma_out = np.concatenate(([sigma_ret], sigma_liv_inc))

    # condition for generator and discriminator
    condition = np.concatenate(
        (
            np.expand_dims(log_rtn_tm1, axis=1),
            np.expand_dims(log_rtn_tm2, axis=1),
            np.expand_dims(realised_vol_tm1, axis=1),
            log_iv_tm1,
        ),
        axis=1,
    )
    # true: what we are trying to predict, increments at time t
    log_rtn_t_ann = np.sqrt(252) * log_rtn_t
    true = np.concatenate((np.expand_dims(log_rtn_t_ann, axis=1), log_iv_inc_t), axis=1)

    return true, condition, m_in, sigma_in, m_out, sigma_out, dates_t, m, tau, ms, taus


def DataTrainValTest(datapath, surfacepath, tr, vl, device="cpu"):
    """
    function to split the data into train, validation, test
    tr and vl are the proportions to use for validation and testing
    """
    true, condition, m_in, sigma_in, m_out, sigma_out, dates_t, m, tau, ms, taus = (
        DataPreprocesssing(datapath, surfacepath)
    )
    data_tt = torch.from_numpy(m_in)
    m_in = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(m_out)
    m_out = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(sigma_in)
    sigma_in = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(sigma_out)
    sigma_out = data_tt.to(torch.float).to(device)
    n = true.shape[0]
    data_tt = torch.from_numpy(true)
    true_tensor = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(condition)
    condition_tensor = data_tt.to(torch.float).to(device)
    true_train = true_tensor[0 : int(tr * n), :]
    true_val = true_tensor[int(tr * n) : int((tr + vl) * n), :]
    true_test = true_tensor[int((tr + vl) * n) :, :]
    condition_train = condition_tensor[0 : int(tr * n), :]
    condition_val = condition_tensor[int(tr * n) : int((tr + vl) * n), :]
    condition_test = condition_tensor[int((tr + vl) * n) :, :]
    return (
        true_train,
        true_val,
        true_test,
        condition_train,
        condition_val,
        condition_test,
        m_in,
        sigma_in,
        m_out,
        sigma_out,
        dates_t,
        m,
        tau,
        ms,
        taus,
    )


def DataTrainTest(datapath, surfacepath, tr, device="cpu"):
    """
    function to split the data into train, test
    tr are the proportions to use for testing
    """
    true, condition, m_in, sigma_in, m_out, sigma_out, dates_t, m, tau, ms, taus = (
        DataPreprocesssing(datapath, surfacepath)
    )
    data_tt = torch.from_numpy(m_in)
    m_in = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(m_out)
    m_out = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(sigma_in)
    sigma_in = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(sigma_out)
    sigma_out = data_tt.to(torch.float).to(device)
    n = true.shape[0]
    data_tt = torch.from_numpy(true)
    true_tensor = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(condition)
    condition_tensor = data_tt.to(torch.float).to(device)
    true_train = true_tensor[0 : int(tr * n), :]
    true_test = true_tensor[int(tr * n) :, :]
    condition_train = condition_tensor[0 : int(tr * n), :]
    condition_test = condition_tensor[int(tr * n) :, :]
    return (
        true_train,
        true_test,
        condition_train,
        condition_test,
        m_in,
        sigma_in,
        m_out,
        sigma_out,
        dates_t,
        m,
        tau,
        ms,
        taus,
    )


def GradientMatching(
    gen,
    gen_opt,
    disc,
    disc_opt,
    criterion,
    condition_train,
    true_train,
    m,
    tau,
    ms,
    taus,
    n_grad,
    lrg,
    lrd,
    batch_size,
    noise_dim,
    device,
    lk=10,
    lt=8,
):
    """
    perform gradient matching
    """
    n_train = condition_train.shape[0]
    n_batches = n_train // batch_size + 1
    dtm = tau * 365
    mP_t, mP_k, mPb_K = penalty_mutau_tensor(m, dtm, device)
    moneyness_t = torch.tensor(m, dtype=torch.float, device=device)

    # Time smoothness: Diffs across time (stride lk)
    # Output size: lk * (lt - 1) = 70
    tau_t = torch.tensor(tau, dtype=torch.float, device=device)
    t_seq = torch.zeros((lt - 1), dtype=torch.float, device=device)
    for i in range(lt - 1):
        t_seq[i] = 1 / ((tau_t[i + 1] - tau_t[i]) ** 2)

    matrix_t = torch.zeros((lk * (lt - 1), lk * lt), device=device, dtype=torch.float)
    for i in range(lk * (lt - 1)):
        matrix_t[i, i] = -1
        matrix_t[i, i + lk] = 1

    # Time weights: [w0...w0 (10), w1...w1 (10)...]
    tsq = torch.repeat_interleave(t_seq, lk).unsqueeze(0)

    # Moneyness smoothness: Diffs across strike (stride 1, skip boundaries)
    # Output size: lt * (lk - 1) = 8 * 9 = 72
    moneyness_t = torch.tensor(m, dtype=torch.float, device=device)
    matrix_m = torch.zeros((lt * (lk - 1), lk * lt), device=device, dtype=torch.float)
    row = 0
    for t_idx in range(lt):
        for k_idx in range(lk - 1):
            col = t_idx * lk + k_idx
            matrix_m[row, col] = -1
            matrix_m[row, col + 1] = 1
            row += 1

    # Moneyness weights: [w0...w8, w0...w8...]
    m_base_seq = torch.zeros((lk - 1), dtype=torch.float, device=device)
    for i in range(lk - 1):
        m_base_seq[i] = 1 / ((moneyness_t[i + 1] - moneyness_t[i]) ** 2)
    m_seq = m_base_seq.repeat(lt).unsqueeze(0)

    n_epochs = n_grad
    discloss = [False] * (n_batches * n_epochs)
    genloss = [False] * (n_batches * n_epochs)
    dscpred_real = [False] * (n_batches * n_epochs)
    dscpred_fake = [False] * (n_batches * n_epochs)
    gen_fake = [False] * (n_batches * n_epochs)
    genprices_fk = [False] * (n_batches * n_epochs)
    BCE_grad = []
    m_smooth_grad = []
    t_smooth_grad = []
    gen.train()
    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(n_train)
        condition_train = condition_train[perm, :]
        true_train = true_train[perm, :]
        for i in range(n_batches):
            curr_batch_size = batch_size
            if i == (n_batches - 1):
                curr_batch_size = n_train - i * batch_size
            condition = condition_train[
                (i * batch_size) : (i * batch_size + curr_batch_size), :
            ]
            surface_past = condition_train[
                (i * batch_size) : (i * batch_size + curr_batch_size), 3:
            ]
            real = true_train[(i * batch_size) : (i * batch_size + curr_batch_size), :]

            real_and_cond = torch.cat((condition, real), dim=-1)
            # update the discriminator
            disc_opt.zero_grad()
            noise = torch.randn(
                (curr_batch_size, noise_dim), device=device, dtype=torch.float
            )
            fake = gen(noise, condition)
            fake_and_cond = torch.cat((condition, fake), dim=-1)

            disc_fake_pred = disc(fake_and_cond.detach())
            disc_real_pred = disc(real_and_cond)
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_opt.step()

            dscpred_real[epoch * n_batches + i] = disc_real_pred[0].detach().item()
            dscpred_fake[epoch * n_batches + i] = disc_fake_pred[0].detach().item()

            discloss[epoch * n_batches + i] = disc_loss.detach().item()

            # update the generator
            gen_opt.zero_grad()
            noise = torch.randn(
                (curr_batch_size, noise_dim), device=device, dtype=torch.float
            )
            fake = gen(noise, condition)

            fake_and_cond = torch.cat((condition, fake), dim=-1)

            disc_fake_pred = disc(fake_and_cond)

            fake_surface = torch.exp(fake[:, 1:] + surface_past)

            penalties_m = [None] * curr_batch_size
            penalties_t = [None] * curr_batch_size
            for iii in range(curr_batch_size):
                penalties_m[iii] = torch.matmul(
                    m_seq, (torch.matmul(matrix_m, fake_surface[iii]) ** 2)
                )
                penalties_t[iii] = torch.matmul(
                    tsq, (torch.matmul(matrix_t, fake_surface[iii]) ** 2)
                )
            m_penalty = sum(penalties_m) / curr_batch_size
            t_penalty = sum(penalties_t) / curr_batch_size

            m_penalty.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            # list of gradient norms
            m_smooth_grad.append(total_norm)
            gen_opt.zero_grad()

            t_penalty.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            # list of gradient norms
            t_smooth_grad.append(total_norm)

            gen_opt.zero_grad()
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            # list of gradient norms
            BCE_grad.append(total_norm)
            gen_opt.step()
            genloss[epoch * n_batches + i] = gen_loss.detach().item()
            gen_fake[epoch * n_batches + i] = fake[0].detach()
            genprices_fk[epoch * n_batches + i] = condition[0].detach()

    alpha = np.mean(np.array(BCE_grad) / np.array(m_smooth_grad))
    beta = np.mean(np.array(BCE_grad) / np.array(t_smooth_grad))
    print("alpha :", alpha, "beta :", beta)
    return gen, gen_opt, disc, disc_opt, criterion, alpha, beta


def GradientMatchingPlot(
    gen,
    gen_opt,
    disc,
    disc_opt,
    criterion,
    condition_train,
    true_train,
    m,
    tau,
    ms,
    taus,
    n_grad,
    lrg,
    lrd,
    batch_size,
    noise_dim,
    device,
    lk=10,
    lt=8,
):
    """
    perform gradient matching and plot
    """
    n_train = condition_train.shape[0]
    n_batches = n_train // batch_size + 1
    dtm = tau * 365
    mP_t, mP_k, mPb_K = penalty_mutau_tensor(m, dtm, device)
    moneyness_t = torch.tensor(m, dtype=torch.float, device=device)

    # Time smoothness: Diffs across time (stride lk)
    # Output size: lk * (lt - 1) = 70
    tau_t = torch.tensor(tau, dtype=torch.float, device=device)
    t_seq = torch.zeros((lt - 1), dtype=torch.float, device=device)
    for i in range(lt - 1):
        t_seq[i] = 1 / ((tau_t[i + 1] - tau_t[i]) ** 2)

    matrix_t = torch.zeros((lk * (lt - 1), lk * lt), device=device, dtype=torch.float)
    for i in range(lk * (lt - 1)):
        matrix_t[i, i] = -1
        matrix_t[i, i + lk] = 1

    # Time weights: [w0...w0 (10), w1...w1 (10)...]
    tsq = torch.repeat_interleave(t_seq, lk).unsqueeze(0)

    # Moneyness smoothness: Diffs across strike (stride 1, skip boundaries)
    # Output size: lt * (lk - 1) = 8 * 9 = 72
    moneyness_t = torch.tensor(m, dtype=torch.float, device=device)
    matrix_m = torch.zeros((lt * (lk - 1), lk * lt), device=device, dtype=torch.float)
    row = 0
    for t_idx in range(lt):
        for k_idx in range(lk - 1):
            col = t_idx * lk + k_idx
            matrix_m[row, col] = -1
            matrix_m[row, col + 1] = 1
            row += 1

    # Moneyness weights: [w0...w8, w0...w8...]
    m_base_seq = torch.zeros((lk - 1), dtype=torch.float, device=device)
    for i in range(lk - 1):
        m_base_seq[i] = 1 / ((moneyness_t[i + 1] - moneyness_t[i]) ** 2)
    m_seq = m_base_seq.repeat(lt).unsqueeze(0)

    n_epochs = n_grad
    discloss = [False] * (n_batches * n_epochs)
    genloss = [False] * (n_batches * n_epochs)
    dscpred_real = [False] * (n_batches * n_epochs)
    dscpred_fake = [False] * (n_batches * n_epochs)
    gen_fake = [False] * (n_batches * n_epochs)
    genprices_fk = [False] * (n_batches * n_epochs)
    BCE_grad = []
    m_smooth_grad = []
    t_smooth_grad = []
    gen.train()
    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(n_train)
        condition_train = condition_train[perm, :]
        true_train = true_train[perm, :]
        for i in range(n_batches):
            curr_batch_size = batch_size
            if i == (n_batches - 1):
                curr_batch_size = n_train - i * batch_size
            condition = condition_train[
                (i * batch_size) : (i * batch_size + curr_batch_size), :
            ]
            surface_past = condition_train[
                (i * batch_size) : (i * batch_size + curr_batch_size), 3:
            ]
            real = true_train[(i * batch_size) : (i * batch_size + curr_batch_size), :]

            real_and_cond = torch.cat((condition, real), dim=-1)
            # update the discriminator
            disc_opt.zero_grad()
            noise = torch.randn(
                (curr_batch_size, noise_dim), device=device, dtype=torch.float
            )
            fake = gen(noise, condition)
            fake_and_cond = torch.cat((condition, fake), dim=-1)

            disc_fake_pred = disc(fake_and_cond.detach())
            disc_real_pred = disc(real_and_cond)
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_opt.step()

            dscpred_real[epoch * n_batches + i] = disc_real_pred[0].detach().item()
            dscpred_fake[epoch * n_batches + i] = disc_fake_pred[0].detach().item()

            discloss[epoch * n_batches + i] = disc_loss.detach().item()

            # update the generator
            gen_opt.zero_grad()
            noise = torch.randn(
                (curr_batch_size, noise_dim), device=device, dtype=torch.float
            )
            fake = gen(noise, condition)

            fake_and_cond = torch.cat((condition, fake), dim=-1)

            disc_fake_pred = disc(fake_and_cond)

            fake_surface = torch.exp(fake[:, 1:] + surface_past)

            penalties_m = [None] * curr_batch_size
            penalties_t = [None] * curr_batch_size
            for iii in range(curr_batch_size):
                penalties_m[iii] = torch.matmul(
                    m_seq, (torch.matmul(matrix_m, fake_surface[iii]) ** 2)
                )
                penalties_t[iii] = torch.matmul(
                    tsq, (torch.matmul(matrix_t, fake_surface[iii]) ** 2)
                )
            m_penalty = sum(penalties_m) / curr_batch_size
            t_penalty = sum(penalties_t) / curr_batch_size

            m_penalty.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            # list of gradient norms
            m_smooth_grad.append(total_norm)
            gen_opt.zero_grad()

            t_penalty.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            # list of gradient norms
            t_smooth_grad.append(total_norm)

            gen_opt.zero_grad()
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            # list of gradient norms
            BCE_grad.append(total_norm)
            gen_opt.step()
            genloss[epoch * n_batches + i] = gen_loss.detach().item()
            gen_fake[epoch * n_batches + i] = fake[0].detach()
            genprices_fk[epoch * n_batches + i] = condition[0].detach()

    alpha = np.mean(np.array(BCE_grad) / np.array(m_smooth_grad))
    beta = np.mean(np.array(BCE_grad) / np.array(t_smooth_grad))

    plt.figure("BCE norm")
    plt.plot(range(len(BCE_grad)), BCE_grad)
    plt.xlabel("iteration")
    plt.ylabel("gradient norm")
    plt.title("BCE gradient norm")
    plt.show()

    plt.figure("Smoothness m")
    plt.plot(range(len(BCE_grad)), m_smooth_grad)
    plt.xlabel("iteration")
    plt.ylabel("gradient norm")
    plt.title("Smoothness penalty for moneyness gradient norm")
    plt.show()

    plt.figure("Smoothness tau")
    plt.plot(range(len(BCE_grad)), t_smooth_grad)
    plt.xlabel("iteration")
    plt.ylabel("gradient norm")
    plt.title("Smoothness penalty for time to maturity gradient norm")
    plt.show()

    print("alpha :", alpha, "beta :", beta)
    return gen, gen_opt, disc, disc_opt, criterion, alpha, beta


def TrainLoopNoVal(
    alpha,
    beta,
    gen,
    gen_opt,
    disc,
    disc_opt,
    criterion,
    condition_train,
    true_train,
    m,
    tau,
    ms,
    taus,
    n_epochs,
    lrg,
    lrd,
    batch_size,
    noise_dim,
    device,
    lk=10,
    lt=8,
):
    """
    train loop for VolGAN
    """
    n_train = condition_train.shape[0]
    n_batches = n_train // batch_size + 1
    dtm = tau * 365
    mP_t, mP_k, mPb_K = penalty_mutau_tensor(m, dtm, device)
    moneyness_t = torch.tensor(m, dtype=torch.float, device=device)
    # Time smoothness: Diffs across time (stride lk)
    # Output size: lk * (lt - 1) = 70
    tau_t = torch.tensor(tau, dtype=torch.float, device=device)
    t_seq = torch.zeros((lt - 1), dtype=torch.float, device=device)
    for i in range(lt - 1):
        t_seq[i] = 1 / ((tau_t[i + 1] - tau_t[i]) ** 2)

    matrix_t = torch.zeros((lk * (lt - 1), lk * lt), device=device, dtype=torch.float)
    for i in range(lk * (lt - 1)):
        matrix_t[i, i] = -1
        matrix_t[i, i + lk] = 1

    # Time weights: [w0...w0 (10), w1...w1 (10)...]
    tsq = torch.repeat_interleave(t_seq, lk).unsqueeze(0)

    # Moneyness smoothness: Diffs across strike (stride 1, skip boundaries)
    # Output size: lt * (lk - 1) = 8 * 9 = 72
    moneyness_t = torch.tensor(m, dtype=torch.float, device=device)
    matrix_m = torch.zeros((lt * (lk - 1), lk * lt), device=device, dtype=torch.float)
    row = 0
    for t_idx in range(lt):
        for k_idx in range(lk - 1):
            col = t_idx * lk + k_idx
            matrix_m[row, col] = -1
            matrix_m[row, col + 1] = 1
            row += 1

    # Moneyness weights: [w0...w8, w0...w8...]
    m_base_seq = torch.zeros((lk - 1), dtype=torch.float, device=device)
    for i in range(lk - 1):
        m_base_seq[i] = 1 / ((moneyness_t[i + 1] - moneyness_t[i]) ** 2)
    m_seq = m_base_seq.repeat(lt).unsqueeze(0)

    discloss = [False] * (n_batches * n_epochs)
    genloss = [False] * (n_batches * n_epochs)
    dscpred_real = [False] * (n_batches * n_epochs)
    dscpred_fake = [False] * (n_batches * n_epochs)
    gen_fake = [False] * (n_batches * n_epochs)
    genprices_fk = [False] * (n_batches * n_epochs)

    gen.train()
    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(n_train)
        condition_train = condition_train[perm, :]
        true_train = true_train[perm, :]
        for i in range(n_batches):
            curr_batch_size = batch_size
            if i == (n_batches - 1):
                curr_batch_size = n_train - i * batch_size
            condition = condition_train[
                (i * batch_size) : (i * batch_size + curr_batch_size), :
            ]
            surface_past = condition_train[
                (i * batch_size) : (i * batch_size + curr_batch_size), 3:
            ]
            real = true_train[(i * batch_size) : (i * batch_size + curr_batch_size), :]

            real_and_cond = torch.cat((condition, real), dim=-1)
            # update the discriminator
            disc_opt.zero_grad()
            noise = torch.randn(
                (curr_batch_size, noise_dim), device=device, dtype=torch.float
            )
            fake = gen(noise, condition)
            fake_and_cond = torch.cat((condition, fake), dim=-1)

            disc_fake_pred = disc(fake_and_cond.detach())
            disc_real_pred = disc(real_and_cond)
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_opt.step()

            dscpred_real[epoch * n_batches + i] = disc_real_pred[0].detach().item()
            dscpred_fake[epoch * n_batches + i] = disc_fake_pred[0].detach().item()

            discloss[epoch * n_batches + i] = disc_loss.detach().item()

            # update the generator
            gen_opt.zero_grad()
            noise = torch.randn(
                (curr_batch_size, noise_dim), device=device, dtype=torch.float
            )
            fake = gen(noise, condition)

            fake_and_cond = torch.cat((condition, fake), dim=-1)

            disc_fake_pred = disc(fake_and_cond)

            # fake_surface = torch.exp(fake[:,1:]+ surface_past)
            fake_surface = fake[:, 1:] + surface_past

            penalties_m = [None] * curr_batch_size
            penalties_t = [None] * curr_batch_size
            for iii in range(curr_batch_size):
                penalties_m[iii] = torch.matmul(
                    m_seq, (torch.matmul(matrix_m, fake_surface[iii]) ** 2)
                )
                penalties_t[iii] = torch.matmul(
                    tsq, (torch.matmul(matrix_t, fake_surface[iii]) ** 2)
                )
            m_penalty = sum(penalties_m) / curr_batch_size
            t_penalty = sum(penalties_t) / curr_batch_size

            gen_opt.zero_grad()
            gen_loss = (
                criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                + alpha * m_penalty
                + beta * t_penalty
            )
            gen_loss.backward()
            gen_opt.step()
            genloss[epoch * n_batches + i] = gen_loss.detach().item()
            gen_fake[epoch * n_batches + i] = fake[0].detach()
            genprices_fk[epoch * n_batches + i] = condition[0].detach()

    return gen, gen_opt, disc, disc_opt, criterion


class VolGANResults(NamedTuple):
    gen: torch.nn.Module
    gen_opt: torch.optim.Optimizer
    disc: torch.nn.Module
    disc_opt: torch.optim.Optimizer
    true_train: np.ndarray
    true_val: np.ndarray
    true_test: np.ndarray
    condition_train: np.ndarray
    condition_val: np.ndarray
    condition_test: np.ndarray
    dates_t: np.ndarray
    m: np.ndarray
    tau: float
    ms: np.ndarray
    taus: np.ndarray


def VolGAN(
    datapath,
    surfacepath,
    tr,
    noise_dim=16,
    hidden_dim=8,
    n_epochs=1000,
    n_grad=100,
    lrg=0.0001,
    lrd=0.0001,
    batch_size=100,
    device="cpu",
):

    (
        true_train,
        true_test,
        condition_train,
        condition_test,
        m_in,
        sigma_in,
        m_out,
        sigma_out,
        dates_t,
        m,
        tau,
        ms,
        taus,
    ) = DataTrainTest(datapath, surfacepath, tr, device)
    gen = Generator(
        noise_dim=noise_dim,
        cond_dim=condition_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=true_train.shape[1],
        mean_in=m_in,
        std_in=sigma_in,
        mean_out=m_out,
        std_out=sigma_out,
    )
    gen.to(device)
    m_disc = torch.cat((m_in, m_out), dim=-1)
    sigma_disc = torch.cat((sigma_in, sigma_out), dim=-1)
    disc = Discriminator(
        in_dim=condition_train.shape[1] + true_train.shape[1],
        hidden_dim=hidden_dim,
        mean=m_disc,
        std=sigma_disc,
    )
    disc.to(device)
    true_val = False
    condition_val = False
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=lrd)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    gen, gen_opt, disc, disc_opt, criterion, alpha, beta = GradientMatching(
        gen,
        gen_opt,
        disc,
        disc_opt,
        criterion,
        condition_train,
        true_train,
        m,
        tau,
        ms,
        taus,
        n_grad,
        lrg,
        lrd,
        batch_size,
        noise_dim,
        device,
    )
    gen, gen_opt, disc, disc_opt, criterion = TrainLoopNoVal(
        alpha,
        beta,
        gen,
        gen_opt,
        disc,
        disc_opt,
        criterion,
        condition_train,
        true_train,
        m,
        tau,
        ms,
        taus,
        n_epochs,
        lrg,
        lrd,
        batch_size,
        noise_dim,
        device,
    )
    return VolGANResults(
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
    )


def reweighting_stats(penalties, beta):
    """
    post re-weighting stats
    penalties: arbitrage penalties #days x #samples
    """
    data_m = penalties
    transform = np.exp(-beta * penalties)
    for i in range(transform.shape[0]):
        transform[i, :] = transform[i, :] / np.sum(transform[i, :])
    mean_before = np.mean(data_m, axis=1)
    median_before = np.median(data_m, axis=1)
    mean_after = np.zeros(transform.shape[0])
    median_after = np.zeros(transform.shape[0])
    for i in tqdm(range(transform.shape[0])):
        mean_after[i] = np.sum(transform[i, :] * data_m[i, :])
        args = np.argsort(data_m[i])
        data_m[i, :] = data_m[i, args]
        transform[i, :] = transform[i, :]
        med = data_m[i, 0]
        sm = 0
        j = 0
        while sm < 0.5:
            sm = sm + transform[i, j]
            med = data_m[i, j]
            j = j + 1
        median_after[i] = med
    print(
        "Mean mean before ",
        np.mean(mean_before),
        " and after reweighting ",
        np.mean(mean_after),
    )
    print(
        "std of means before ",
        np.std(mean_before),
        " and after reweighting across time ",
        np.std(mean_after),
    )
    print(
        "Mean median before ",
        np.median(mean_before),
        " and median of means",
        np.median(mean_after),
    )

    plt.hist(
        data_m[1, :],
        bins=50,
        density=True,
        weights=transform[1, :],
        color="blue",
        label="Reweighted data",
    )
    plt.hist(data_m[1, :], bins=50, density=True, color="red", label="Original data")
    plt.legend(loc="upper center")
    plt.title("Arbitrage violations of generated data on a sample day")
    plt.show()
    plt.hist(mean_after, bins=50, density=True, color="blue", label="Reweighted data")
    plt.hist(mean_before, bins=50, density=True, color="red", label="Original data")
    plt.legend(loc="upper right")
    plt.title("Mean mean arbitrage violations of generated data")
    plt.show()
    plt.hist(median_after, bins=50, density=True, color="blue", label="Reweighted data")
    plt.hist(median_before, bins=50, density=True, color="red", label="Original data")
    plt.legend(loc="upper right")
    plt.title("Mean medians of arbitrage violations of generated data")
    plt.show()
    return data_m, transform


def reweighting_stats_mean(penalties, beta):
    """
    post re-weighting stats (mean)
    penalties: arbitrage penalties #days x #samples
    """
    data_m = penalties
    transform = np.exp(-beta * data_m)
    for i in range(transform.shape[0]):
        transform[i, :] = transform[i, :] / np.sum(transform[i, :])

    mean_after = np.zeros(transform.shape[0])
    for i in tqdm(range(transform.shape[0])):
        mean_after[i] = np.sum(transform[i, :] * data_m[i, :])
    median_after = np.zeros(transform.shape[0])
    for i in tqdm(range(transform.shape[0])):
        mean_after[i] = np.sum(transform[i, :] * data_m[i, :])
        args = np.argsort(data_m[i])
        data_m[i, :] = data_m[i, args]
        transform[i, :] = transform[i, :]
        med = data_m[i, 0]
        sm = 0
        j = 0
        while sm < 0.5:
            sm = sm + transform[i, j]
            med = data_m[i, j]
            j = j + 1
        median_after[i] = med

    return mean_after, median_after


def VolGAN_sample(sample_surfaces, sample_returns, weights):
    """
    sample_surfaces, sample_returns: raw generated
    weights: associated with the raw samples
    """
    n = weights.shape[0]
    N = weights.shape[1]
    Nm = sample_surfaces.shape[2]
    sample_back_s = np.zeros((n, Nm))
    sample_back_r = np.zeros(n)
    for i in range(n):
        index = np.random.choice(range(N), size=1, replace=True, p=weights[i, :])
        sample_back_s[i, :] = sample_surfaces[index, i, :]
        sample_back_r[i] = sample_returns[index, i]
    return sample_back_s, sample_back_r


def VolGAN_mean_surface_day(variable, weights):
    """
    mean surface for a ficed day (VolGAN)
    """
    mn = np.zeros(variable.shape[1])
    for i in range(len(weights)):
        mn = mn + weights[i] * variable[i, :]
    return mn


def VolGAN_quantile_surface_day(variable, weights, quantile):
    """
    VolGAN quantile for a fixed day
    """
    args = np.argsort(weights)
    weights = weights[args]
    variable = variable[args, :]
    tot = 0
    i = 0
    while tot < quantile:
        tot = tot + weights[i]
        i = i + 1
    if i == len(weights):
        qnt = variable[-1, :]
    else:
        qnt = variable[i, :]
    return qnt


def VolGAN_mean_ret_day(variable, weights):
    """
    mean VolGAN return on a fixed day
    """
    mn = 0
    for i in range(len(weights)):
        mn = mn + weights[i] * variable[i]
    return mn


def VolGAN_quantile_ret_day(variable, weights, quantile):
    """
    returns VolGAN quantile on a fixed day
    """
    args = np.argsort(variable)
    weights = weights[args]
    variable = variable[args]
    tot = 0
    i = 0
    while tot < quantile:
        tot = tot + weights[i]
        i = i + 1
    if i == len(weights):
        qnt = variable[-1]
    else:
        qnt = variable[i]
    return qnt


def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, array(xs))))

    return ufunclike


def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    y = np.interp(x, xp, yp)
    y = np.where(x < xp[0], yp[0] + (x - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1]), y)
    y = np.where(
        x > xp[-1], yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]), y
    )
    return y


def get_points3(I_known, m_known, tau_known, m_want, tau_want):
    """
    given I(K,T) returns points in I(K*,tau*) using interpolation in K and T
    K_known and T_known need to be sequences
    """
    # interpolate Ks first (fix tau known)
    xvals = m_want
    S = np.zeros(shape=(len(m_want), len(tau_known)))
    for i in range(len(tau_known)):
        yinterp = extrap(xvals, m_known, I_known[:, i])
        S[:, i] = yinterp
    # interpolate taus now
    xvals = tau_want
    I = np.zeros(shape=(len(m_want), len(tau_want)))
    xvals = tau_want
    for i in range(len(m_want)):
        yinterp = extrap(xvals, tau_known, S[i, :])
        I[i, :] = yinterp
    return I


def VIX(St, IV, m, tau):
    """
    VIX simulation
    St: underlying
    IV: implied vol surface
    m, tau: moneyness and implied vol grid
    """
    m_min = 0.5
    m_max = 1.5
    n_s = 50
    m1 = np.linspace(m_min, m_max, 2 * n_s)
    strikes = m1 * St
    i0 = np.argmin(np.abs(strikes - St))
    sigma1 = get_points3(IV, m, tau, m1, [1 / 12])[0][0]
    r = 0
    Call = BS_OptionPrice(St, 1 / 12, r, strikes, sigma1)
    Put = Call - St + strikes
    DeltaK = np.zeros(len(strikes))
    DeltaK[1:] = np.diff(strikes)
    DeltaK[0] = DeltaK[1]
    VIX = 0
    for i in range(i0 + 1):
        VIX = VIX + Put[i] * DeltaK[i] / (strikes[i] ** 2)
    VIX = 100 * np.sqrt(2 / (1 / 12) * VIX - 1 / (1 / 12) * (St / strikes[i0] - 1) ** 2)
    return VIX


def quantile(quantity, weights, quantile):
    """
    custom quantile function
    """
    indxs = np.argsort(quantity)
    quantity = quantity[indxs]
    weights = weights[indxs]
    i = -1
    w = 0
    while w < quantile:
        i = i + 1
        w = w + weights[i]
    return quantity[i]
