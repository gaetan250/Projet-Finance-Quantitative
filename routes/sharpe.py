import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# 🔹 Charger les données de volatilité prédite et de rendements
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

# 🔹 Optimisation du portefeuille avec maximisation du ratio de Sharpe
def optimize_portfolio_sharpe():
    log_returns, volatility_predicted_ma = load_data()
    
    vol_dict = {crypto: volatility_predicted_ma[crypto] for crypto in log_returns.columns if crypto in volatility_predicted_ma}
    
    last_30_returns = log_returns.iloc[-30:]
    mu = last_30_returns.mean().values 

    volatilities = np.array(list(vol_dict.values()))
    corr_matrix = last_30_returns.corr().values  
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

    # 🔹 Fonction de maximisation du ratio de Sharpe
    def sharpe_ratio(weights, mu, cov_matrix, risk_free_rate=0):
        portfolio_return = np.dot(weights, mu) 
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  
        return -(portfolio_return - risk_free_rate) / portfolio_volatility  

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  
    bounds = [(-1,1) for _ in range(len(mu))]  
    
    initial_weights = np.random.uniform(-1, 1, size=len(mu))
    initial_weights /= np.sum(np.abs(initial_weights)) 
    
    result = minimize(sharpe_ratio, initial_weights, args=(mu, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    max_sharpe_ratio = -result.fun
    
    return optimal_weights, max_sharpe_ratio, vol_dict, mu, last_30_returns

# 🔹 Graphique des Rendements Cumulés
def plot_cumulative_returns(optimal_weights, last_30_returns):
    portfolio_cumulative_returns = (1 + np.dot(last_30_returns, optimal_weights)).cumprod()
    plt.figure(figsize=(10, 5))
    plt.plot(last_30_returns.index, portfolio_cumulative_returns, label="Optimized Portfolio", color='red')
    plt.xlabel("Date")
    plt.ylabel("Cumulative performance")
    plt.title("Sharpe-based optimized portfolio performance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("static/sharpe_cumulative_returns.png")
    plt.close()

# 🔹 Exécution principale
def generate_plots_and_response(request: Request):
    optimal_weights, max_sharpe_ratio, vol_dict, mu, last_30_returns = optimize_portfolio_sharpe()
    
    # 📊 Graphique des performances cumulées
    plot_cumulative_returns(optimal_weights, last_30_returns)
    
    # 📊 Graphique de la répartition des poids
    plt.figure(figsize=(10, 5))
    plt.bar(list(vol_dict.keys()), optimal_weights, color='lightcoral')
    plt.xlabel("Cryptos")
    plt.ylabel("Poids optimaux")
    plt.title("Répartition Optimale du Portefeuille (Max Sharpe)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("static/sharpe_portfolio_weights.png")
    plt.close()
    
    return templates.TemplateResponse("sharpe_results.html", {
        "request": request,
        "sharpe_ratio": max_sharpe_ratio,
        "optimal_weights": list(optimal_weights),  # Convertir en liste explicite
        "assets": list(vol_dict.keys()),  # Convertir en liste explicite
        "zip": zip  # Passer zip comme variable au template
    })


@router.get("/run")
def run_portfolio_optimization(request: Request):
    return generate_plots_and_response(request)
