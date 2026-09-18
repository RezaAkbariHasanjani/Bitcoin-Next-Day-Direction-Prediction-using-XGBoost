# Bitcoin-Next-Day-Direction-Prediction-using-XGBoost
An XGBoost-based machine learning project for next-day Bitcoin direction prediction using historical OHLCV data, engineered features, and TimeSeriesSplit cross-validation.


# Bitcoin Next-Day Direction Prediction Using XGBoost

## Overview

This project investigates whether historical Bitcoin market data can be used to predict the **direction of the next day's closing price** using the **XGBoost** machine learning algorithm.

The project uses daily OHLCV data obtained from Yahoo Finance and applies feature engineering, chronological train-test splitting, and time-series cross-validation to evaluate the model's ability to generalize to unseen future observations.

> **Important:** This project predicts the **direction of the next day's closing price**, not the exact Bitcoin price.

---

## Objective

The main objective is to build and evaluate a binary classification model that predicts whether Bitcoin's closing price will increase or decrease on the following day.

The target variable is defined as:

* `1` → Tomorrow's closing price is higher than today's closing price
* `0` → Tomorrow's closing price is not higher than today's closing price

---

## Dataset

Bitcoin daily market data were downloaded using the `yfinance` Python library.

**Asset:** BTC-USD
**Period:** January 1, 2020 – September 15, 2026
**Frequency:** Daily

The original market variables include:

* Open
* High
* Low
* Close
* Volume

---

## Feature Engineering

In addition to the original OHLCV variables, several derived features were created:

| Feature         | Description                               |
| --------------- | ----------------------------------------- |
| `benefit`       | Difference between Close and Open         |
| `Return`        | Daily percentage return                   |
| `HL_range`      | High-Low range normalized by Close        |
| `OC_range`      | Open-Close range normalized by Open       |
| `Volume_Change` | Daily percentage change in trading volume |

The final feature set consists of:

```text
Open
High
Low
Close
Volume
benefit
Return
HL_range
OC_range
Volume_Change
```

---

## Target Definition

The target variable represents the direction of the following day's closing price.

```python
df["Tomorrow"] = df["Close"].shift(-1)

df["Target"] = (
    df["Tomorrow"] > df["Close"]
).astype(int)
```

Therefore:

```text
Target = 1 → Price increases the next day
Target = 0 → Price does not increase the next day
```

---

## Methodology

The project follows a chronological machine learning workflow.

### 1. Data Collection

Bitcoin data are downloaded using `yfinance`.

### 2. Data Preprocessing

The dataset is converted to a pandas DataFrame, dates are converted to datetime format, and missing observations generated during feature engineering are removed.

### 3. Feature Engineering

Market-derived features such as daily return, price ranges, and volume changes are calculated.

### 4. Chronological Train-Test Split

The dataset is divided without shuffling:

```python
train_test_split(
    X,
    y,
    test_size=0.1,
    shuffle=False
)
```

This ensures that the test set represents a later period than the training data.

### 5. Hyperparameter Optimization

`GridSearchCV` is combined with `TimeSeriesSplit` using five folds.

A total of **384 hyperparameter combinations** were evaluated:

```text
384 candidates × 5 folds
= 1,920 model fits
```

The optimization metric was classification accuracy.

---

## XGBoost Model

The final model was selected using time-series cross-validation.

### Best Hyperparameters

```text
n_estimators      = 200
max_depth         = 3
learning_rate     = 0.1
subsample         = 0.5
colsample_bytree  = 0.8
reg_alpha         = 0.1
reg_lambda        = 1
```

---

## Results

### Training Performance

The optimized model achieved:

```text
Accuracy: 75.79%
```

Classification performance:

| Class                | Precision | Recall | F1-score |
| -------------------- | --------: | -----: | -------: |
| 0                    |      0.76 |   0.74 |     0.75 |
| 1                    |      0.76 |   0.78 |     0.77 |
| **Overall Accuracy** |           |        | **0.76** |

### Test Performance

On the unseen future test period:

```text
Test Accuracy: 50.20%
```

Classification performance:

| Class                | Precision | Recall | F1-score |
| -------------------- | --------: | -----: | -------: |
| 0                    |      0.52 |   0.60 |     0.56 |
| 1                    |      0.47 |   0.39 |     0.43 |
| **Overall Accuracy** |           |        | **0.50** |

### Test Confusion Matrix

```text
              Predicted
              0     1
Actual  0    77    51
        1    71    46
```

---

## Baseline Comparison

The test set contains:

```text
Class 0: 128 samples
Class 1: 117 samples
```

A simple majority-class classifier that always predicts class `0` would achieve:

```text
128 / 245 ≈ 52.24%
```

The XGBoost model achieved:

```text
50.20%
```

on the same test period.

Therefore, the model did not outperform the simple majority-class baseline on the evaluated unseen period.

---

## Interpretation

The substantial difference between training and test performance is an important finding.

```text
Training Accuracy
       ↓
     75.79%
       ↓
Unseen Test Period
       ↓
     50.20%
```

The results indicate that the model does not generalize well to the future Bitcoin observations used for testing.

This suggests that the selected historical OHLCV-derived features may not contain sufficient stable predictive information for reliable next-day Bitcoin direction prediction over the evaluated period.

The result also demonstrates why chronological evaluation is important when working with financial time-series data.

---

## Limitations

Several limitations should be considered:

* The model uses a relatively small set of market-derived features.
* Bitcoin price movements are affected by many external factors that are not represented in the dataset.
* The model does not incorporate news, sentiment, macroeconomic variables, blockchain metrics, or order-book information.
* Accuracy alone may not be sufficient for evaluating a trading strategy.
* Transaction costs, slippage, and risk management are not modeled.
* The test period represents only one future period and therefore cannot establish general performance across all market regimes.

---

## Future Work

Potential extensions of this project include:

* Comparing XGBoost with Logistic Regression, Random Forest, SVM, and other classifiers.
* Adding technical indicators such as RSI, MACD, ATR, and moving averages.
* Investigating lagged features.
* Evaluating feature importance and model interpretability.
* Testing different prediction horizons.
* Comparing classification metrics such as ROC-AUC, F1-score, precision, and recall.
* Investigating class probability calibration and decision thresholds.
* Evaluating model performance across different Bitcoin market regimes.
* Incorporating additional information such as market sentiment and macroeconomic variables.

---

## Technologies

The project was developed using Python and the following libraries:

```text
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost
yfinance
```

---


## Disclaimer

This project is intended for **educational and research purposes**.

The model is not a financial advisory system and the reported results should not be interpreted as evidence that Bitcoin price movements can be reliably predicted or that the model can generate profitable trading strategies.

---

## Author

**Dr.Reza Akbari-Hasanjani**

Electronic Engineering | Machine Learning | AI | Data Analysis

GitHub: `RezaAkbariHasanjani`
