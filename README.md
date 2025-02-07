# 📌 Project: Volatility Prediction & Portfolio Optimization

This project aims to **predict cryptocurrency volatility** using an **LSTM-GARCH-CVI** model and then **optimize a portfolio** by maximizing the **Sharpe ratio** and **Conditional Value at Risk (CVaR)**.

The entire system is structured with **FastAPI**, enabling access to functionalities via an API.

---

## **📂 Project Structure**
```
📁Finance-Quantitave-Project
│── 📂 routes
│   │── data_processing.py   # Data fetching (Crypto & CVI)
│   │── models_garch.py      # GARCH volatility model
│   │── models_lstm.py       # LSTM-GARCH-CVI volatility model
│   │── sharpe.py            # Portfolio optimization with Sharpe ratio
│   │── sharpe_cvar.py       # Portfolio optimization with CVaR
│── 📂 templates
│   │── index.html           # API web interface
│   │── fetch_data.html      # Data visualization
│   │── garch_results.html   # GARCH model results
│   │── lstm_results.html    # LSTM-GARCH-CVI model results
│   │── sharpe_results.html  # Sharpe ratio optimization results
│   │── sharpe_cvar_results.html  # CVaR optimization results
│── 📂 static
│   │── styles.css           # Web interface styling
│   │── plots (Generated graphs)
│── main.py                  # Main FastAPI entry point
│── README.md                # Complete documentation
│── requirements.txt          # Required dependencies
```

---

## **🚀 Installation and Setup**
### **1️⃣ Install Dependencies**
Ensure you have **Python 3.8+**, then run:
```sh
pip install -r requirements.txt
```

### **2️⃣ Start the FastAPI Server**
```sh
uvicorn main:app --reload
```

---

## **🛠️ Features and Endpoints**
### **1️⃣ Data Fetching**
| **Endpoint** | **Description** |
|-------------|---------------|
| `/fetch-data/run` | Downloads cryptocurrency prices (`yfinance`) and the CVI index (`Investing.com`). |
| `/fetch-data/plot` | Generates a chart of cryptocurrency prices. |

---

### **2️⃣ GARCH Model**
| **Endpoint** | **Description** |
|-------------|---------------|
| `/garch/train-garch` | Trains a GARCH model for cryptocurrencies and saves volatility predictions. |
| `/garch/plot-garch` | Displays a comparison between actual and predicted volatility from GARCH. |

---

### **3️⃣ LSTM-GARCH-CVI Model**
| **Endpoint** | **Description** |
|-------------|---------------|
| `/lstm-garch-cvi/run` | Trains the LSTM model and generates volatility predictions. |
| `/lstm-garch-cvi/plot` | Displays a comparison between actual and predicted volatility. |

---

### **4️⃣ Portfolio Optimization**
#### 🔹 **Classic Sharpe Ratio**
| **Endpoint** | **Description** |
|-------------|---------------|
| `/sharpe/run` | Optimizes the portfolio using the Sharpe ratio based on LSTM-GARCH-CVI volatility. |

#### 🔹 **CVaR-Based Optimization**
| **Endpoint** | **Description** |
|-------------|---------------|
| `/sharpe-cvar/run` | Optimizes the portfolio by maximizing the Sharpe ratio based on CVaR. |

---

### **5️⃣ Server Health and Documentation**
| **Endpoint** | **Description** |
|-------------|---------------|
| `/health` | Checks server health and the presence of critical files (`log_returns.csv`, `crypto_prices.csv`, `cvi_data.csv`). |
| `/readme` | Displays this README directly from the API. |

---

## **📈 Model Explanations**
### **1️⃣ GARCH Model**
- Used to model conditional volatility.
- Expresses variance based on past shocks.
- Implemented with **`arch`**.

### **2️⃣ LSTM-GARCH-CVI Model**
- **Inputs**: Historical volatility, GARCH volatility, CVI.
- **Recurrent neural network (LSTM)** captures temporal patterns.
- **Optimized** with `Adam` and `MSE` loss function.

### **3️⃣ Portfolio Optimization**
#### 📌 **Sharpe Ratio**
- Maximization of:
  \[
  Sharpe = \frac{E[R_p] - R_f}{\sigma_p}
  \]
- Where \(E[R_p]\) = expected return, \(R_f\) = risk-free rate, \(\sigma_p\) = portfolio volatility.

#### 📌 **CVaR (Conditional Value at Risk)**
- Reduces extreme risk by optimizing:
  \[
  CVaR_{\alpha} = E[R | R < VaR_{\alpha}]
  \]
- Optimization **minimizes potential loss**.

---

## **📊 Results and Visualization**
- **Graphs available** in `static/`
- Comparison of **actual vs. predicted volatility**
- Distribution of **optimized portfolio weights**
- Cumulative performance of **Sharpe & CVaR portfolios**

---

## **🔗 Usage Examples**
### **1️⃣ Test the Endpoints**
Open [Swagger UI](http://127.0.0.1:8000/docs):
```sh
http://127.0.0.1:8000/docs
```

### **2️⃣ Run Predictions**
```sh
curl -X GET "http://127.0.0.1:8000/lstm-garch-cvi/run"
```

### **3️⃣ Optimize a Portfolio**
```sh
curl -X GET "http://127.0.0.1:8000/sharpe/run"
```

---

## **🛠 Technologies Used**
- **FastAPI** 🚀 (Ultra-fast API framework)
- **NumPy / Pandas** 📊 (Data manipulation)
- **Matplotlib** 📈 (Visualization)
- **TensorFlow / Keras** 🤖 (Deep Learning LSTM)
- **ARCH** 📉 (GARCH modeling)
- **SciPy** 🏗 (Mathematical optimization)

---

## **📝 Future Improvements**
✔️ Add **portfolio backtesting**.  
✔️ Include **interactive visualization** via **Streamlit**.  
✔️ Integrate **bidirectional RNNs** for better predictions.  

---

## **👨‍💻 Author**
- **MOSEF Tonin Rivory Gaétan Dumas Pierre Liberge**
- **Quantitative Finance Project**
- **Submission Date: February 15, 2024**

---

This should be **perfectly clear and comprehensive**! 😊🔥 If you need any additions, let me know! 🚀

