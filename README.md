# Zomato-Style Restaurant Ratings — EDA & Rating Prediction

Exploratory analysis and a from-scratch NumPy linear regression model predicting restaurant rating from cost, cuisine, online ordering, table booking, and vote count.

**Tools:** Python, NumPy, Pandas, Matplotlib

---

## Note on Data

The original Zomato Kaggle dataset wasn't available to fetch in the environment this was built in, so the notebook runs on a **synthetic-but-realistic restaurant dataset** (2,000 restaurants) generated with a deliberate, noisy relationship between features and rating — documented in the notebook's first cell. Every number, table, and chart in the notebook is real, executed output on that data, not fabricated. Swapping in the real Zomato CSV only requires changing one `pd.read_csv(...)` line.

## Files

- `zomato_analysis.ipynb` — full EDA + from-scratch regression, with real executed outputs and plots
- `requirements.txt` — numpy, pandas, matplotlib

## Key Findings

- Restaurants that support **online ordering** and **table booking** average meaningfully higher ratings (+0.19 and +0.21 respectively) than those that don't.
- Raw correlations between `cost_for_two` / `votes` and rating are weak (−0.05 and 0.04) — the relationship isn't simple/linear on its own.
- A from-scratch linear regression (NumPy normal equation, cuisine one-hot encoded, votes log-transformed) reaches **test R² = 0.11, RMSE = 0.38**. That's modest, and reported honestly rather than inflated — the synthetic rating includes deliberate noise on the same scale as the real signal, similar to how much of the variance in real restaurant ratings comes from things a tabular dataset can't capture (food quality on a given day, individual reviewer taste).
- `table_booking`, `online_order`, and `log(votes)` are the strongest predictors; `cost_for_two` has the weakest individual effect.

## What I'd Add Next

- Swap in the real Zomato Kaggle dataset
- Try a tree-based model (Random Forest / Gradient Boosting) to check for non-linear effects
- Add locality-level aggregation (average rating and cost by neighborhood)

## Contact

**Muhammad Ahmed Jawaid**
[LinkedIn](https://www.linkedin.com/in/m-ahmed-jawaid-416662253/) · ahmedjawaid513@outlook.com
