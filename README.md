# 📈 Stock ML Correlation & Regression Dashboard

An end-to-end **machine learning analysis pipeline** that studies the relationship between stock prices and financial fundamentals for major Indian IT companies using multiple regression models, cross-validation, and a professional multi-panel dashboard.

This project demonstrates **ML experimentation, feature importance analysis, correlation handling, and results visualization** in a single, reproducible workflow.

---

## 🚀 Project Objective

The goal of this project is to:

- Analyze how **financial fundamentals** relate to **stock price movements**
- Compare the performance of **8 different regression models**
- Identify the **top 3 most influential variables per company**
- Present results in a **clean, executive-style dashboard**

This mirrors real-world workflows used in **equity research, quantitative analysis, and data science roles**.

---

## 🏢 Companies Analyzed

- TCS  
- Infosys (INFY)  
- HCL Technologies  
- Wipro  
- Tech Mahindra (TECHM)

---

## 📊 Fundamental Variables Used

- Sales Growth  
- EBITDA Margin Change  
- EBITDA Growth  
- PAT Growth  
- PAT Margin Change  

> ⚠️ Note:  
> To maintain a controlled ML experiment and ensure sufficient sample size alignment with daily stock prices, **fundamental features are generated using trend-based statistical simulations**.  
> This allows consistent comparison across models while preserving realistic financial behavior.

---

## 🧠 Machine Learning Models Implemented

The pipeline evaluates the following models using **R² score**:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- ElasticNet  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Support Vector Regression (RBF kernel)

Additionally:
- **Random Forest cross-validation** is performed (3-fold CV)
- Feature importance is aggregated across models

---

## 🔍 Key Techniques & Engineering Highlights

- Safe correlation computation with NaN and zero-variance protection  
- Feature scaling using `StandardScaler`  
- Train–test split with reproducibility  
- Cross-model feature importance averaging  
- Defensive programming for model compatibility  
- Professional multi-panel visualization layout  

---

## 📈 Dashboard Output (6 Panels)

1. **Correlation Heatmap**  
   Stock price vs financial fundamentals (per company)

2. **Model Performance Heatmap**  
   R² comparison across all ML models

3. **Best Model Comparison**  
   Highest-performing model per company

4. **Cross-Validation Scores**  
   Random Forest CV performance

5. **Top Variable Distribution**  
   Most influential fundamental per company

6. **Stock Price Trends**  
   Time-series visualization for selected stocks

---

## 🛠 Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Data Handling | pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualization | matplotlib, seaborn |
| Modeling | Regression, Ensembles, SVR |
| Environment | Local / Jupyter / Script |

---

## 📁 Project Structure

