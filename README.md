# InsightX: Pharmaceutical Drug Sales Demand Prediction

Predicting daily drug sales with exploratory data analysis and machine learning, to support stock planning in a pharmacy.

## Highlights

- **365 days** of daily sales data (4 drugs, 4 categories, 2023)
- Full workflow: data checks, EDA, feature engineering, time-based split, model training, evaluation, interpretation, demand planning
- Two models compared against a naive baseline: **Linear Regression** and a tuned **Random Forest Regressor**
- Best model on the test period: **Linear Regression**, MAE **6.7 units/day**, R-squared **0.51** (the naive baseline has MAE 11.6 and R-squared -0.32)
- A **demand-planning table** (predicted daily demand per category and month) turns the model into a practical tool

> **Note on the data:** `data.csv` is **synthetic**. It keeps the columns of the original notebook (`Date`, `Drug_ID`, `Category`, `Quantity_Sold`, `Unit_Price`), but `Quantity_Sold` now follows realistic patterns (seasonality, weekend dip, price effect) plus noise. In the original version the quantity was purely random; on the same split, Linear Regression and Random Forest then score R-squared of about -0.01 and -0.18, i.e. nothing can be learned. The scores below demonstrate the workflow, not real-world accuracy.

## Repository structure

| File | Description |
|------|-------------|
| `Drug_Sales_Demand_Prediction.ipynb` | Complete notebook (EDA, features, models, evaluation, importance, demand planning) |
| `InsightX_Project_Report.pdf` | Full project report |
| `data.csv` | Synthetic daily sales dataset (365 rows x 5 columns) |
| `linear_regression_model.pkl` | Trained Linear Regression pipeline (preprocessing + model) |
| `random_forest_model.pkl` | Trained and tuned Random Forest pipeline (preprocessing + model) |
| `README.md` | This file |

## Dataset

| Column | Description |
|--------|-------------|
| `Date` | Day of sale (1 Jan to 31 Dec 2023) |
| `Drug_ID` | `Drug_A` ... `Drug_D` |
| `Category` | Antibiotics, Analgesics, Antivirals, Cardiology |
| `Quantity_Sold` | **Target:** units sold that day |
| `Unit_Price` | Selling price per unit |

## Methodology

1. **EDA** - distributions, averages by category and drug, monthly seasonality, weekday and price effects, correlations, revenue
2. **Feature engineering** - `Month`, `Quarter`, `DayOfWeek`, `Is_Weekend`, plus cyclical `Month_Sin` / `Month_Cos`
3. **Preprocessing** - one-hot encoding of `Drug_ID` and `Category`, MinMax scaling of `Unit_Price` (inside a scikit-learn `Pipeline`)
4. **Time-based split** - first 80% of days for training (292 days), last 20% for testing (73 days, 2023-10-20 to 2023-12-31); no shuffling
5. **Models** - naive baseline, Linear Regression, Random Forest (grid search with time-series cross-validation; best: `n_estimators=300`, `max_depth=8`, `min_samples_leaf=1`)
6. **Evaluation** - MAE, RMSE, R-squared, MAPE
7. **Interpretation** - permutation importance, regression coefficients, demand-planning table

## Results (test period, 73 days)

| Model | MAE (units) | RMSE (units) | R-squared | MAPE (%) |
|-------|------------:|-------------:|----------:|---------:|
| Naive baseline (mean) | 11.60 | 13.85 | -0.319 | 23.9 |
| Linear Regression | 6.68 | 8.48 | 0.505 | 13.4 |
| Random Forest (tuned) | 7.67 | 9.70 | 0.354 | 15.3 |

**Key insights**

- Both models beat the naive baseline; Linear Regression is best. The Random Forest overfits (train R-squared 0.93 vs test 0.35) because only 292 training days are available.
- Strongest demand drivers: `Unit_Price`, `Drug_ID`, `Category` (permutation importance); `Unit_Price` dominates.
- Antibiotics and Antivirals peak in winter; weekends sell less; higher prices reduce demand.

## How to run

**Google Colab**

1. Open `Drug_Sales_Demand_Prediction.ipynb` in Colab (File > Open notebook > GitHub).
2. Upload `data.csv` to the Colab file panel (if it is missing, the notebook generates a new synthetic copy).
3. Run all cells.

**Locally**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Drug_Sales_Demand_Prediction.ipynb
```

## Using the saved models

The `.pkl` files are full pipelines, so raw features go in and a predicted quantity comes out.

```python
import pickle
import numpy as np
import pandas as pd

with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

def make_scenario(date, drug, category, price):
    d = pd.Timestamp(date)
    return {"Drug_ID": drug, "Category": category, "Month": d.month, "DayOfWeek": d.dayofweek,
            "Is_Weekend": int(d.dayofweek >= 5), "Quarter": d.quarter,
            "Month_Sin": np.sin(2 * np.pi * (d.month - 1) / 12),
            "Month_Cos": np.cos(2 * np.pi * (d.month - 1) / 12),
            "Unit_Price": price}

new = pd.DataFrame([make_scenario("2024-01-16", "Drug_A", "Antibiotics", 40.0)])
new = new[list(model.feature_names_in_)]
print("Predicted units:", round(model.predict(new)[0], 1))
```

> The models were created with scikit-learn 1.8.0. If another version raises an error when loading a `.pkl` file, re-run the notebook to regenerate them.

## Limitations and future work

- Synthetic, single-year data; validate on several years of real sales before drawing real-world conclusions.
- Each day has one sale record, with `Drug_ID` and `Category` assigned independently; real data would have one row per drug per day.
- No holidays, promotions, stock-outs or weather information.
- Next steps: lag features, gradient boosting, SARIMA / Prophet, forecast intervals, and a dashboard.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
