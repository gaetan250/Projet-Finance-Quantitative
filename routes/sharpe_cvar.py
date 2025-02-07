import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def load_data():
    log_returns = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)
    
    volatility_predictions = {}
    for crypto in log_returns.columns:
        vol_path = f"results/lstm_garch_cvi_predictions_{crypto}.csv"
        if os.path.exists(vol_path):
            vol_pred = pd.read_csv(vol_path, index_col=0, parse_dates=True)
            volatility_predictions[crypto] = vol_pred["Test_Predictions"].dropna().iloc[-30:].mean()
        else:
            raise FileNotFoundError(f"🚨 Erreur : Fichier {vol_path} non trouvé ! Veuillez d'abord exécuter l'endpoint 'Prédire Volatilité LSTM GARCH CVI' et réessayer.")
    
    return log_returns, volatility_predictions

def portfolio_cvar(weights, portfolio_returns, alpha=0.05):
    portf_returns = np.dot(portfolio_returns, weights)
    var = np.percentile(portf_returns, alpha * 100)
    cvar = np.mean(portf_returns[portf_returns <= var])  
    return abs(cvar)  

def sharpe_ratio_cvar(weights, mu, portfolio_returns, risk_free_rate=0.03/365):
    portf_return = np.dot(weights, mu) 
    portf_cvar = portfolio_cvar(weights, portfolio_returns)
    return -(portf_return - risk_free_rate) / portf_cvar  

def optimize_portfolio_cvar():
    log_returns, volatility_predicted_ma = load_data()
    
    vol_dict = {crypto: volatility_predicted_ma[crypto] for crypto in log_returns.columns if crypto in volatility_predicted_ma}
    
    last_30_returns = log_returns.iloc[-30:]
    mu_cvar = last_30_returns.mean().values  
    volatilities_cvar = np.array(list(vol_dict.values()))
    corr_matrix_cvar = last_30_returns.corr().values  
    cov_matrix_cvar = np.outer(volatilities_cvar, volatilities_cvar) * corr_matrix_cvar  

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(-1,1) for _ in range(len(mu_cvar))] 

    initial_weights = np.random.uniform(-1, 1, size=len(mu_cvar))
    initial_weights /= np.sum(np.abs(initial_weights))  

    opt_result_cvar = minimize(
        sharpe_ratio_cvar,
        initial_weights,
        args=(mu_cvar, last_30_returns),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights_cvar = opt_result_cvar.x
    max_sharpe_ratio_cvar = -opt_result_cvar.fun

    return optimal_weights_cvar, max_sharpe_ratio_cvar, vol_dict, mu_cvar, last_30_returns

def plot_cumulative_returns_cvar(optimal_weights_cvar, last_30_returns):
    portfolio_cumulative_returns_cvar = (1 + np.dot(last_30_returns, optimal_weights_cvar)).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(last_30_returns.index, portfolio_cumulative_returns_cvar, label="Optimized Portfolio (CVaR)", color='red')
    plt.xlabel("Date")
    plt.ylabel("Cumulative performance")
    plt.title("Sharpe-based optimized portfolio performance based on CVaR")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("static/sharpe_cvar_cumulative_returns.png")
    plt.close()

def generate_plots_and_response(request: Request):
    optimal_weights_cvar, max_sharpe_ratio_cvar, vol_dict, mu_cvar, last_30_returns = optimize_portfolio_cvar()

    plot_cumulative_returns_cvar(optimal_weights_cvar, last_30_returns)

    plt.figure(figsize=(10, 5))
    plt.bar(list(vol_dict.keys()), optimal_weights_cvar, color='lightcoral')
    plt.xlabel("Cryptos")
    plt.ylabel("Poids optimaux")
    plt.title("Répartition Optimale du Portefeuille (Max Sharpe basé sur la CVaR)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("static/sharpe_cvar_portfolio_weights.png")
    plt.close()

    return templates.TemplateResponse("sharpe_cvar_results.html", {
        "request": request,
        "sharpe_ratio": max_sharpe_ratio_cvar,
        "optimal_weights": list(optimal_weights_cvar),
        "assets": list(vol_dict.keys()),
        "zip": zip
    })

@router.get("/run")
def run_portfolio_optimization(request: Request):
    return generate_plots_and_response(request)
