# Supply Chain Analytics: Delivery Lead Time Prediction with XGBoost & GenAI

This repository contains an end‑to‑end **predictive supply chain** framework that transforms DataCo’s logistics from reactive planning to an intelligence‑driven system for predicting delivery lead times and late‑delivery risk.[file:2][file:4][file:5]  
The project combines rigorous data cleaning, feature engineering, gradient‑boosting models, SHAP‑based explainability, and a GenAI layer that converts model outputs into operational actions.[file:3][file:4][file:6]

---

## 1. Project Overview

The core business problem is uncertainty in promised delivery dates: traditional heuristic models exhibit an average error of about **1.28 days**, which undermines customer trust and forces high safety buffers in inventory and capacity planning.[file:2][file:4][file:5]  
Using more than **180,000 shipment records** and 15+ engineered features, this project builds a production‑ready XGBoost model that reduces **Mean Absolute Error (MAE) to 0.88 days**, representing roughly **31% improvement** over the baseline and enabling significantly more reliable delivery commitments.[file:2][file:3][file:4][file:5]

### Key Objectives

- Predict **delivery lead time** and **late_delivery_risk (0 = on‑time, 1 = late)** before shipment.[file:3][file:5]  
- Identify the main **operational drivers** of delay (e.g., shipping mode, scheduling, routes).[[file:3]]  
- Provide **explainable AI** outputs using SHAP to ensure the model behaves consistently with logistics domain logic.[file:4][file:5][file:6]  
- Integrate a **GenAI communication layer** that turns numerical predictions and SHAP values into alerts, root‑cause explanations, and executive summaries.[file:2][file:4]

---

## 2. Dataset & Problem Definition

### 2.1 Source and Scale

The dataset is a structured supply‑chain table capturing full order lifecycles: order placement, shipment processing, delivery timestamps, and associated customer, product, and route attributes.[file:3][file:5]  
After cleaning, the final modeling data includes approximately **180,519 to 300,000 shipment records** (depending on the specific modeling version) and around **15 predictive features** plus 25 engineered attributes.[file:3][file:4][file:5]

### 2.2 Target Variables

Two related targets are used in the pipeline:

- `days_for_shipping_real`: continuous **actual delivery lead time** in days (0–6 days), used for exploratory analysis and regression benchmarks.[file:3][file:5]  
- `late_delivery_risk`: **binary classification label** where 1 indicates a delivery that exceeded the scheduled duration, and 0 represents on‑time performance.[file:3][file:5]

The **primary modeling objective** for deployment is the **binary late_delivery_risk** because it is more directly actionable for proactive logistics decisions.[file:3][file:5]

### 2.3 Feature Groups

Features fall into four broad groups.[file:3][file:5]

- Temporal: order date, shipment date, derived day‑of‑week, month, and order hour.  
- Geographic: customer state, origin–destination zones, geospatial distance proxies.  
- Operational: shipping mode, planned shipment duration, logistics routes, holiday indicators.  
- Order/financial: product category, quantity, revenue, profit, and discount fields.

EDA shows that **operational and scheduling variables dominate delivery performance**, while financial metrics and IDs have negligible correlation with lead time.[file:3][file:5]

---

## 3. Data Quality, Cleaning & EDA

### 3.1 Data Integrity

A structured quality process confirms that the dataset is logically consistent and modeling‑ready.[file:3][file:5]

- No duplicate shipment records after identifier checks.  
- **0% missing values** across modeling features except `product_description`, which is 100% missing and dropped.[file:3]  
- Timestamp validation ensures **no “delivery before order” cases** and lead times are naturally bounded between 0 and 6 days.[file:3][file:5]  
- No extreme outliers beyond the 95th percentile; thus, no capping or record removal is required.[file:3]

### 3.2 Target Behaviour

For `days_for_shipping_real`:[file:3]

- Mean ≈ 3.50 days, median = 3 days, IQR = 2–5 days.  
- Slight right skew but overall compact distribution within operational expectations.  

For `late_delivery_risk`:[file:3]

- Late (1): ~54.8% of orders; On‑time (0): ~45.2%.  
- Mild class imbalance (≈1.2:1) handled via stratified sampling and class weighting.[file:3]

### 3.3 Key EDA Insights

- **Shipping mode** is a strong driver: Same‑day has the lowest lead times, Standard Class the highest and most variable.[file:3][file:5]  
- Product category, payment mode, and most destination states have **minimal impact** on lead time.[file:3][file:5]  
- Temporal patterns are stable across weekdays and months; orders placed late in the day and during holidays suffer slightly longer lead times (~0.1 day difference).[file:3]  
- Geospatial analysis shows moderate variation across major destination states, while thousands of very sparse routes (<50 records) require special handling to avoid overfitting.[file:3]

---

## 4. Feature Engineering & Leakage Prevention

The pipeline introduces **25+ engineered features** to better represent logistics behaviour and to support high‑capacity models like XGBoost.[file:2][file:3][file:4]

### 4.1 Engineered Features (Examples)

- **Delivery Speed Ratio**: ratio of actual lead time to scheduled duration.[file:4]  
- **Geospatial Zones**: origin and destination grouped into operational zones to capture distance and route complexity.[file:2][file:3][file:4]  
- Temporal flags: day of week, month, hour bucket, and holiday indicator.[file:3]  
- Aggregated route labels: sparse routes with <50 observations grouped into “Other_Route” to stabilize estimates.[file:3]

### 4.2 Leakage Control

To ensure the model cannot “cheat” by using information that is only available after delivery, several columns are explicitly excluded:[file:3][file:4]

- `days_for_shipping_real` (actual lead time).  
- `delivery_status` and any variables derived from post‑delivery outcomes.  

Only variables known at **order creation or shipment scheduling time** are used for training.[file:3][file:4]

### 4.3 Encoding & Splitting

- Categorical variables (shipping mode, market/region, payment mode, product category) are **one‑hot encoded**.[file:3][file:4]  
- High‑cardinality states are split into top‑frequency states plus an “Other_State” bucket.[file:3]  
- Data is split into **train, validation, and test sets**, preserving class balance through **stratified sampling**; time‑based splits are recommended for production to respect temporal structure.[file:3][file:6]

---

## 5. Modeling Approach

Multiple model families are implemented and benchmarked, covering both regression and classification views of the problem.[file:4][file:5][file:6]

### 5.1 Models Implemented

- Heuristic baseline model using simple historical averages or rules.[file:4][file:6]  
- ARIMA time‑series baseline for comparison, though its structure is not well aligned to shipment‑level data.[file:4][file:6]  
- Classical ML: Logistic Regression for classification of late deliveries.[file:4][file:5]  
- Tree‑based ensembles: **Random Forest**, **XGBoost**, **LightGBM**.[file:4][file:5][file:6]  
- **Stacked Ensemble** meta‑model combining multiple base learners to maximize performance.[file:4][file:6]

For each model, both **regression metrics** (MAE, RMSE, R²) and **classification metrics** (Accuracy, F1‑score) are calculated, while training time is recorded for deployment trade‑offs.[file:4][file:6]

### 5.2 Evaluation & Benchmarks

A consolidated comparison table highlights the trade‑offs between different architectures.[file:4][file:5]

| Model                | MAE (days) | F1‑Score | Notes |
|----------------------|-----------:|:--------:|------|
| Heuristic Baseline   | ~1.27–1.28 | ~0.58    | Simple average/heuristic rules.[file:4][file:5] |
| Logistic Regression  | ~1.15      | ~0.70    | Strong linear baseline.[file:4][file:5] |
| **XGBoost (Selected)** | **0.88**   | **0.85** | Best balance of accuracy and speed.[file:2][file:4] |
| Stacked Ensemble     | ~0.84      | ~0.87    | Slightly better metrics, higher complexity.[file:4][file:6] |

Residual plots, confusion matrices, and calibration curves show that gradient‑boosting models achieve **well‑calibrated probabilities** and errors centered around zero, with strong ability to flag delayed shipments.[file:4][file:5][file:6]

### 5.3 Final Model Choice

Although the stacked ensemble provides marginally lower MAE, **XGBoost** is selected as the **production model** because it delivers almost identical performance with substantially lower training and inference time.[file:4][file:6]  
On the held‑out validation data, XGBoost reaches **MAE ≈ 0.88 days**, correctly classifies about **85% of late deliveries**, and attains **ROC‑AUC ≈ 0.88**, demonstrating robust separation between late and on‑time shipments across routes.[file:2][file:4]

---

## 6. Explainability with SHAP

To support “open‑box AI,” SHAP analysis is applied to the XGBoost model so that planners and leadership can understand why each prediction is made.[file:2][file:4][file:5][file:6]

### 6.1 Global Feature Importance

SHAP summary plots and feature‑importance charts highlight the dominant drivers of late‑delivery risk.[file:3][file:4][file:5]

- **Shipping Mode** (e.g., air vs. road vs. standard class) is the single most impactful feature on predicted delay.[file:3][file:4]  
- **Geospatial distance proxies** and **origin/destination zones** are the next strongest contributors, reflecting the role of route length and network structure.[file:3][file:4]  
- Temporal patterns (days since order, order hour, holiday flag) have smaller but still meaningful influence.[file:3]  
- Financial metrics such as revenue, discount, and profit per order show minimal SHAP contribution, confirming that price does not drive lead time in this dataset.[file:3][file:5]

### 6.2 Local Explanations

For individual shipments, SHAP values explain whether risk is driven by long‑haul standard shipping, peak‑hour order placement, or particular high‑risk routes.[file:4][file:5]  
These explanations can be surfaced directly to operations teams and used by GenAI to auto‑generate plain‑language root‑cause narratives.[file:2][file:4]

---

## 7. GenAI Operational Integration

Beyond pure modeling, the project proposes a **three‑step GenAI workflow** that converts predictive insights into business actions.[file:2][file:4]

1. **Predictive Model Layer**  
   - XGBoost generates lead‑time estimates and late‑delivery probabilities for each shipment.[file:4][file:5]  

2. **GenAI Interpreter Layer**  
   - A Large Language Model ingests raw predictions and SHAP explanations to produce:  
     - Risk alerts for operations, highlighting shipments likely to be late and the operational causes.[file:2][file:4]  
     - Automated **root‑cause emails** to internal stakeholders, translating “0.88 days predicted delay” into clear, polite updates about rerouting or prioritization.[file:2][file:4]  
     - Weekly PDF **executive summaries** summarizing model health, delay hotspots, and improvement opportunities.[file:2][file:4]

3. **Action & Monitoring Layer**  
   - Output can be connected to dashboards, messaging systems, or ticketing tools to trigger interventions on high‑risk shipments and track improvement over time.[file:4][file:5]

This architecture ensures that model outputs do not remain siloed and instead directly influence routing decisions, buffer stock policies, and customer communication.[file:2][file:4][file:5]

---

## 8. Business Impact

By combining accurate prediction with explainable AI and a GenAI communication layer, the system delivers tangible business value.[file:2][file:4][file:5]

- **31% reduction** in average delivery‑time prediction error vs. heuristic baselines, enabling more precise ETAs.[file:2][file:4]  
- Ability to **flag ~85% of true late deliveries** early enough for proactive intervention, improving customer satisfaction and reducing penalty costs.[file:4][file:5]  
- Potential **10–15% reduction in buffer stock** through tighter safety‑time and inventory policies informed by accurate risk forecasts.[file:2][file:4]  
- Data‑driven insights that reveal bottlenecks in Standard Class long‑haul routes, guiding targeted process and carrier improvements.[file:3][file:4][file:5]

Overall, the project demonstrates how predictive analytics can convert a traditional supply chain into a proactive, **intelligence‑driven logistics network**.[file:4][file:5]

---

## 9. Repository Structure

> The exact folder names may vary; this structure reflects the logical organization of the project based on the attached files.

.
├── data/
│   └── raw/
│   |   └── DataCoSupplyChainDataset.csv          # Original dataset (as used in notebooks)
│   ├── processed/
│   │   ├── final_cleaned_supply_chain_dataset.csv
│   │   ├── initial_cleaned_supply_chain_dataset.csv
│   │   └── data_quality_summaries.csv
├── notebooks/
│   ├── data_cleaning-2.ipynb                     # Data integrity checks & cleaning
│   ├── feature_engineering-3.ipynb               # Feature engineering + train/val/test creation
│   └── SHAP_Supply_Chain_FSR_Version.ipynb       # Modeling, evaluation & SHAP analysis
├── reports/
│   ├── EDA-FINDINGS-2.pdf                        # EDA summary
│   ├── SUPPLY-CHAIN-ANALYTICS-REPORT-1-.pdf-5.pdf# Full project report
│   ├── SCA-Modeling-Evaluation-supply-chain-project_-3.pdf  # Modeling review
│   ├── Supply-Chain-Analytics-1-6.pdf            # Slide deck (PDF)
│   ├── Speaker_Notes-4.pdf                       # Speaker notes
│   └── slides/
│       └── Supply-chain-project.pptx.pptx        # Original PPTX presentation
└── README.md                                     # GitHub project documentation
