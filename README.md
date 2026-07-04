# SwiftCart Demand Forecasting Pipeline

An end-to-end ML pipeline for inventory demand forecasting across 50 dark stores of SwiftCart, an Indian retail chain. Built over 5 weeks as part of a data science internship, each notebook layers a new capability on top of the last — from raw EDA to cost-aware, localized forecasting.

---

## Business Context

SwiftCart operates 50 stores across 5 SKU categories: **Produce, Dairy, Bakery, Pantry, and Snacks**. Manual procurement by store managers was causing measurable revenue loss through a combination of overstocking (spoilage) and understocking (stockouts). The asymmetric cost structure — Rs 1 per spoiled unit vs Rs 10 per stockout unit — means that a symmetric ML model is not enough; the forecasting system must be cost-aware.

---

## Key Results

| Metric | Value |
|--------|-------|
| Revenue bleed from manual procurement | Rs 64,577 annualised (11.07%) |
| Best model WAPE | 16.08% (Global Random Forest) |
| Best Total Business Cost | Rs 8,30,842 (Cost-Sensitive RF + 25% buffer) |
| Saving vs manager baseline | **Rs 30,073 (3.5% reduction)** |
| Manager baseline TBC | Rs 8,60,915 |

---

## Project Structure

```
swiftcart/
├── week1.ipynb                    # EDA & business problem framing
├── week2.ipynb                    # Feature engineering
├── week3.ipynb                    # Baseline ML models
├── week4.ipynb                    # Asymmetric cost optimization
├── week5.ipynb                    # Localized store-level forecasting
├── dataset.csv                    # Raw synthetic sales data
└── dataset_week2_engineered.csv   # Engineered feature dataset (89,500 rows)
```

---

## Week-by-Week Breakdown

### Week 1 — Exploratory Data Analysis (`week1.ipynb`)
- Profiled demand patterns across all 50 stores and 5 SKU categories
- Identified and quantified the **11.07% revenue bleed** (Rs 64,577 annualised) from manual procurement errors
- Established the business case for a data-driven forecasting system

### Week 2 — Feature Engineering (`week2.ipynb`)
- Built three feature groups on top of raw sales data:
  - **Temporal:** day of week, season, public holiday flags
  - **Weather:** temperature, rainfall, humidity
  - **Store-location:** population density, household income, zone classification
- Constructed `Actual_Units_Demanded = Units_Sold + Est_Lost_Sales` as the uncensored demand target — avoids the bias introduced when `Units_Sold` is capped by the manager's order quantity during stockouts
- Output: `dataset_week2_engineered.csv` (89,500 rows)

### Week 3 — Baseline ML Models (`week3.ipynb`)
- Implemented and compared Ridge Regression, Decision Tree, and Random Forest on the engineered dataset
- Train/test split: Jan–Sep 2023 (train) / Oct–Dec 2023 (test)
- Lag features (Lag-1, Lag-7, 3-day rolling average) computed from uncensored demand
- Random Forest emerged as the strongest baseline

### Week 4 — Asymmetric Cost Optimization (`week4.ipynb`)
- Introduced the **Total Business Cost (TBC)** metric: `TBC = spoiled × Rs1 + stockout × Rs10`
- Applied two asymmetric levers:
  1. **Cost-sensitive sample weights** at training time (10× weight on stockout records)
  2. **Safety buffer sweep** at inference time: `Order = Prediction × (1 + B)`
- Optimal buffer **B\* = 25%** minimises TBC at **Rs 8,30,842** — beating the manager baseline by Rs 30,073
- Trained a `RandomForestClassifier` for stockout risk scoring; ROC-AUC ≈ 0.497 (near-random), confirming `Stockout_Flag` is driven by manager ordering behaviour, not demand features — so targeted buffering reduces to a uniform flat multiplier
- 5% spoilage constraint is not achievable in this dataset (minimum ~8%), consistent with the manager's own 7.2% rate

### Week 5 — Localized Store-Level Forecasting (`week5.ipynb`)
- Benchmarked two localization strategies against the global RF baseline (WAPE 16.08%):
  - **Strategy 1:** Global LightGBM with target-encoded `Store_ID` — WAPE 16.47%
  - **Strategy 2:** 50 independent store micro-models (LightGBM) — WAPE 18.17%
- Both strategies underperformed the baseline; root cause identified: all 50 stores have near-identical mean demand (CV = 0.57%), leaving no inter-store signal for localization to exploit
- The methodology is correct — in a real dataset with genuine store heterogeneity (different zones, foot traffic, demographics), both strategies would deliver measurable lift
- **Global RF remains the recommended production model**

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn` (Ridge, Decision Tree, Random Forest) |
| Gradient boosting | `LightGBM` |
| Visualisation | `matplotlib`, `seaborn` |
| Environment | Python 3.x, Jupyter Notebook |

---

## Setup

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

Clone the repo and run notebooks in order (`week1` → `week5`). All notebooks read from `dataset_week2_engineered.csv` — run `week2.ipynb` first if the file is not present.

---

## Key Takeaways

- **Uncensored demand matters.** Using `Units_Sold` as the target silently underfits because stockouts cap observed sales below true demand. The correct target is `Units_Sold + Est_Lost_Sales`.
- **Symmetric loss is wrong for asymmetric costs.** A standard MSE model treats a 10-unit overstock and a 10-unit stockout identically — a Rs 90 error in this business. Sample weights and safety buffers are practical levers to correct this.
- **Localization needs variance to work.** Store micro-models and target encoding only help when stores are meaningfully different. Diagnosing heterogeneity (CV of store means) before investing in localization is essential.
- **Dataset limitations are findings too.** Clearly quantifying what the data cannot support (5% spoilage constraint, inter-store localization) is as valuable as the results themselves.

---

*Internship project — SwiftCart Analytics | SOJOL DAS*
