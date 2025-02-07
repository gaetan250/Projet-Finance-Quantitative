import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import yfinance as yf

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# 📌 Définition des fichiers
LOG_RETURNS_FILE = "log_returns.csv"
CRYPTO_PRICES_FILE = "crypto_prices.csv"

# 📌 Paramètres optimaux des modèles GARCH
best_garch_params = {
    "ADA-USD": {"p": 4, "q": 3, "constant": True},
    "BNB-USD": {"p": 2, "q": 3, "constant": True},
    "BTC-USD": {"p": 4, "q": 4, "constant": True},
    "ETH-USD": {"p": 4, "q": 3, "constant": True},
}

def fetch_crypto_prices():
    """Télécharge les prix des cryptos depuis Yahoo Finance si nécessaire."""
    cryptos = list(best_garch_params.keys())
    start_date = "2019-03-11"
    end_date = "2024-11-28"

    if not os.path.exists(CRYPTO_PRICES_FILE):
        print("📥 Téléchargement des prix des cryptos...")
        try:
            data = yf.download(cryptos, start=start_date, end=end_date, interval='1d')['Close']
            data.to_csv(CRYPTO_PRICES_FILE)
            print(f"✅ Crypto prices saved in {CRYPTO_PRICES_FILE}.")
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement des prix des cryptos : {e}")
            return None
    else:
        print("🔄 Fichier crypto_prices.csv déjà disponible.")

    return pd.read_csv(CRYPTO_PRICES_FILE, index_col=0, parse_dates=True)

def load_or_generate_log_returns():
    """ Vérifie si log_returns.csv existe, sinon le génère depuis crypto_prices.csv. """
    if not os.path.exists(LOG_RETURNS_FILE):
        print("📥 Génération du fichier log_returns.csv...")

        prices = fetch_crypto_prices()
        if prices is None:
            print("❌ Impossible de générer log_returns.csv car les prix des cryptos sont indisponibles.")
            return None

        log_returns = np.log(prices / prices.shift(1)).dropna()
        log_returns.to_csv(LOG_RETURNS_FILE)
        print(f"✅ Fichier log_returns.csv créé avec succès.")

    return pd.read_csv(LOG_RETURNS_FILE, index_col=0, parse_dates=True)

def train_garch_with_forecast(crypto_name):
    """ Entraîne un modèle GARCH et génère des prédictions. """
    log_returns = load_or_generate_log_returns()

    if log_returns is None or crypto_name not in log_returns.columns:
        return {"error": f"Données indisponibles pour {crypto_name}. Vérifiez la création du fichier log_returns.csv."}

    if crypto_name not in best_garch_params:
        return {"error": f"Pas de paramètres GARCH définis pour {crypto_name}."}

    params = best_garch_params[crypto_name]
    p, q, constant = params["p"], params["q"], params["constant"]
    crypto_returns = log_returns[crypto_name].dropna()

    model = arch_model(crypto_returns, vol="Garch", p=p, q=q, mean="Constant" if constant else "Zero")
    results = model.fit(disp="off")

    os.makedirs("results", exist_ok=True)
    volatility_df = pd.DataFrame({"Date": crypto_returns.index, "Volatility": results.conditional_volatility.values})
    volatility_df.to_csv(f"results/garch_volatility_{crypto_name}.csv", index=False)

    return {"message": f"Modèle GARCH entraîné pour {crypto_name} et prédictions sauvegardées."}

@router.get("/train-garch")
def train_all_garch():
    """ Entraîne GARCH sur toutes les cryptos et sauvegarde les prédictions. """
    for crypto in best_garch_params.keys():
        train_garch_with_forecast(crypto)

    return {"message": "Tous les modèles GARCH ont été entraînés et sauvegardés."}

@router.get("/plot-garch", response_class=HTMLResponse)
def plot_garch_volatility(request: Request):
    """ Génère un graphique comparant la volatilité réalisée et prédite. """
    log_returns = load_or_generate_log_returns()

    if log_returns is None:
        return {"error": "Les données log_returns.csv sont manquantes. Exécutez d'abord /train-garch."}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    metrics_summary = []

    for i, crypto in enumerate(best_garch_params.keys()):
        try:
            print(f"📌 Traitement de {crypto}...")

            predicted_volatility = pd.read_csv(f"results/garch_volatility_{crypto}.csv", parse_dates=["Date"])
            realized_volatility = log_returns[crypto].rolling(window=30).std().dropna()

            print(f"✅ {crypto} - Avant alignement : Réalisée {realized_volatility.shape}, Prédite {predicted_volatility.shape}")

            predicted_volatility.set_index("Date", inplace=True)
            realized_volatility.fillna(method="ffill", inplace=True)
            predicted_volatility.fillna(method="ffill", inplace=True)

            realized_volatility, predicted_volatility_aligned = realized_volatility.align(predicted_volatility["Volatility"], join="inner")

            print(f"✅ {crypto} - Après alignement : Réalisée {realized_volatility.shape}, Prédite {predicted_volatility_aligned.shape}")

            if realized_volatility.empty or predicted_volatility_aligned.empty:
                print(f"⚠️ {crypto} - Problème : séries vides après alignement.")
                continue

            mse = mean_squared_error(realized_volatility, predicted_volatility_aligned)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(realized_volatility, predicted_volatility_aligned)
            metrics_summary.append({"Crypto": crypto, "MSE": mse, "RMSE": rmse, "MAE": mae})

            ax = axes[i]
            ax.plot(realized_volatility.index, realized_volatility, label="Realized Volatility", color="blue")
            ax.plot(predicted_volatility_aligned.index, predicted_volatility_aligned, label="Predicted Volatility", color="red")
            ax.set_title(f"Volatility Comparison for {crypto}")
            ax.legend()
            ax.grid(True)

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {crypto}: {e}")

    plt.tight_layout()
    plt.savefig("static/garch_plot.png")
    plt.close()

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_html = metrics_df.to_html(classes="table table-striped", index=False)

    return templates.TemplateResponse("garch_results.html", {"request": request, "metrics_html": metrics_html})

@router.get("/run")
def run_garch(request: Request):
    """ Exécute GARCH et affiche les résultats """
    return plot_garch_volatility(request)
