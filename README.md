# 📌 Research Paper Overview & Execution Guide

## **🔍 Overview**
This project explores **cryptocurrency volatility prediction and portfolio optimization** using advanced machine learning and econometric models. The methodology integrates **LSTM (Long Short-Term Memory) networks**, **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**, and **CVI (Crypto Volatility Index)** for improved financial risk assessment and investment decision-making.

The objective is to optimize portfolio allocation by leveraging **Sharpe Ratio Maximization** and **Conditional Value at Risk (CVaR)** techniques, ensuring **risk-adjusted returns** are optimized.

---

## **📖 Paper Contributions**
### **1️⃣ Volatility Prediction using LSTM-GARCH-CVI**
- **GARCH models** are used to estimate the time-varying volatility of cryptocurrencies.
- **LSTM models** leverage deep learning to capture long-term dependencies and patterns in financial time series data.
- **Crypto Volatility Index (CVI)** is included to enhance predictive power by incorporating overall market sentiment and risk perception.

### **2️⃣ Portfolio Optimization**
- **Classic Sharpe Ratio Optimization:**
  - Maximizes the Sharpe Ratio:
    \[
    Sharpe = \frac{E[R_p] - R_f}{\sigma_p}
    \]
  - Where \(E[R_p]\) is expected return, \(R_f\) is the risk-free rate, and \(\sigma_p\) is portfolio volatility.
- **CVaR-Based Optimization:**
  - Focuses on **tail risk management**, ensuring protection against extreme losses.
  - Uses:
    \[
    CVaR_{\alpha} = E[R | R < VaR_{\alpha}]
    \]
  - Which represents expected loss given that the portfolio is in the worst \(\alpha\)% of cases.

### **3️⃣ Practical Implementation via API**
- The entire methodology is encapsulated within a **FastAPI-based microservice**, allowing users to:
  - **Train and deploy LSTM-GARCH models**
  - **Fetch real-time market data**
  - **Optimize portfolios dynamically**
  - **Visualize volatility predictions and allocation strategies**

---

## **🛠️ Execution Guide**
### **1️⃣ Install Dependencies**
Before running the code, install all required dependencies:
```sh
pip install -r requirements.txt
```

### **2️⃣ Execute the Research Notebook**
Run the **Jupyter Notebook** provided to train models, generate predictions, and perform initial analyses:
```sh
jupyter FQ_NB.ipynb
```

### **3️⃣ Start the API Server**
To launch the FastAPI-based server, execute the following command:
```sh
uvicorn main:app --reload
```

This will start the server at `http://127.0.0.1:8000`, where you can access:
- **API Documentation**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/health`
- **Endpoints for fetching data, predictions, and optimization and so on**

---

## **📈 Results & Findings**
- **Improved volatility predictions** using a hybrid deep learning + econometric approach.
- **Enhanced portfolio returns** by integrating deep learning forecasts into allocation decisions.
- **Robust risk management** via **CVaR optimization**, reducing downside risk significantly.
- **Real-time deployable API**, making volatility forecasting and portfolio optimization accessible.

---

## **📌 Final Notes**
This project offers a **practical framework** for applying advanced **machine learning and financial econometrics** in cryptocurrency markets. By combining **LSTM, GARCH, and CVI**, it provides more **accurate volatility predictions**, leading to **smarter investment decisions**.


