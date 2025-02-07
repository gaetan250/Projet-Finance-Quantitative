### 📌 **API for Cryptocurrency Volatility Prediction & Portfolio Optimization**

This FastAPI-based project provides an **API** for:
- **Fetching cryptocurrency and CVI (Crypto Volatility Index) data**
- **Predicting volatility using LSTM-GARCH-CVI models**
- **Optimizing a portfolio using the Sharpe ratio and CVaR (Conditional Value at Risk)**

---

## **📂 Project Structure**
```
📁 Projet-Finance-Quantitative--API
│️─️ 📂 routes
│   │️─️ data_processing.py   # Data fetching (Crypto & CVI)
│   │️─️ models_garch.py      # GARCH volatility model
│   │️─️ models_lstm.py       # LSTM-GARCH-CVI volatility model
│   │️─️ sharpe.py            # Portfolio optimization (Sharpe ratio)
│   │️─️ sharpe_cvar.py       # Portfolio optimization (CVaR)
│️─️ 📂 templates
│   │️─️ index.html           # API web interface
│   │️─️ fetch_data.html      # Data visualization
│   │️─️ garch_results.html   # GARCH model results
│   │️─️ lstm_results.html    # LSTM-GARCH-CVI model results
│   │️─️ sharpe_results.html  # Sharpe optimization results
│   │️─️ sharpe_cvar_results.html  # CVaR optimization results
│️─️ 📂 static
│   │️─️ styles.css           # CSS for frontend
│   │️─️ all image in format.png # Auto-generatedcharts
│️─️ main.py                  # FastAPI entry point
│️─️ README.md                # Documentation
│️─️ requirements.txt          # Dependencies
```

---

## **🚀 Installation and Setup**
### **1️⃣ Install Dependencies**
Ensure you have **Python 3.8+**, then install the required packages:
```sh
pip install -r requirements.txt
```

### **2️⃣ Start the API Server**
Run the following command:
```sh
uvicorn main:app --reload
```
This will start the API at:
```
http://127.0.0.1:8000
```

### **3️⃣ API Documentation**
You can explore and test the API directly using:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Redoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## **🛠️ API Endpoints**
### **1️⃣ Data Fetching**
| **Endpoint** | **Method** | **Description** |
|-------------|-----------|----------------|
| `/fetch-data/run` | **GET** | Downloads cryptocurrency prices (`yfinance`) and CVI index (`Investing.com`). |
| `/fetch-data/plot` | **GET** | Generates a chart of cryptocurrency prices. |

---

### **2️⃣ GARCH Volatility Model**
| **Endpoint** | **Method** | **Description** |
|-------------|-----------|----------------|
| `/garch/train-garch` | **GET** | Trains a GARCH model for each cryptocurrency. |
| `/garch/plot-garch` | **GET** | Displays actual vs. predicted volatility using GARCH. |

---

### **3️⃣ LSTM  Volatility Model**
| **Endpoint** | **Method** | **Description** |
|-------------|-----------|----------------|
| `/lstm/run` | **GET** | Trains the LSTM-GARCH-CVI model and generates volatility predictions. |
| `/lstm/plot` | **GET** | Visualizes actual vs. predicted volatility. |

---

### **4️⃣ Portfolio Optimization**
#### **📌 Classic Sharpe Ratio Optimization**
| **Endpoint** | **Method** | **Description** |
|-------------|-----------|----------------|
| `/sharpe/run` | **GET** | Optimizes a portfolio using the **Sharpe ratio** based on LSTM-GARCH-CVI volatility. |

#### **📌 CVaR-Based Portfolio Optimization**
| **Endpoint** | **Method** | **Description** |
|-------------|-----------|----------------|
| `/sharpe-cvar/run` | **GET** | Optimizes the portfolio based on **Conditional Value at Risk (CVaR)**. |

---

### **5️⃣ Server Health and Documentation**
| **Endpoint** | **Method** | **Description** |
|-------------|-----------|----------------|
| `/health` | **GET** | Checks if the server is running and verifies essential files. |
| `/readme` | **GET** | Returns this README.md file content as an API response. |

