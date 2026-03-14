# 🛰️ GeoInsights Analytics — Earth Observation AI Platform

## 📋 Project Overview
This project is a comprehensive solution for Case Study 89: **Earth Observation Data Analytics**. It analyzes the business feasibility and technical implementation of an AI-powered satellite imagery platform.

### Key Components:
1.  **Streamlit Dashboard (`app.py`):** An interactive, real-time application to explore financial projections, EO maps (NDVI, LST), ML model performance, and CRM metrics.
2.  **Jupyter Notebook (`eo_analytics_case_study.ipynb`):** A detailed research notebook containing the core algorithms, data simulations, and static visualizations.
3.  **Simulation Engine:** Custom Python code using `numpy` and `scikit-learn` to generate synthetic satellite data and train a Land Cover Classification model.

---

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
To launch the interactive Streamlit app:
```bash
streamlit run app.py
```

### 3. Open the Notebook
To view the research and data analysis:
```bash
jupyter notebook eo_analytics_case_study.ipynb
```

---

## 👨‍🏫 Panel Presentation Prep: Frequently Asked Questions

### 1. Financial & Business Model (Q1 & Q2)
*   **Q: Why is the Year-1 Net Benefit negative?**
    *   **A:** The upfront investment is ₹18 Crores (covering infra, data access, and AI development), while Year-1 revenue is ₹14 Crores. This results in an initial deficit of ₹10 Crores (after operating costs of ₹6 Crores).
*   **Q: When does the business become profitable?**
    *   **A:** Based on default parameters (15% growth, 4L fee), break-even is achieved in **Year 2**. By Year 5, the cumulative profit is significant.
*   **Q: How sensitive is the business to Churn Rate?**
    *   **A:** High. In the CRM section of the app, you can see that increasing churn significantly reduces **Customer Lifetime Value (CLV)**. Keeping churn below 10-15% is critical for long-term survival.

### 2. Earth Observation Analytics (Q3 & Q4)
*   **Q: What is NDVI and why is it used?**
    *   **A:** NDVI (Normalized Difference Vegetation Index) measures vegetation health using NIR and Red bands. It's the "gold standard" for Agricultural clients to monitor crop stress and yield.
*   **Q: How did you simulate Land Surface Temperature (LST)?**
    *   **A:** LST was simulated using `numpy` by creating a correlation with NDVI (Urban Heat Island effect). Areas with low vegetation (Urban) generally show higher temperatures, while dense forests show lower temperatures.
*   **Q: Why is 'Processing Time' a critical KPI?**
    *   **A:** Satellite data is massive. If it takes hours to process a single scene, the platform cannot scale. We set an SLA target of **< 30 seconds per scene** to ensure a high-quality user experience.

### 3. Machine Learning & Accuracy
*   **Q: Which algorithm did you use for Land Cover Classification?**
    *   **A:** **Random Forest Classifier**. It is excellent for multi-class tabular satellite data, handles non-linear relationships well, and provides "Feature Importance" to show which bands (like NIR) were most useful.
*   **Q: How do you handle "Cloud Cover"?**
    *   **A:** Cloud cover is a "noise" variable. In a real-world system, we would use a mask to filter out clouds before processing. In the simulation, we track it as a Data Quality metric.

### 4. CRM & Growth (Q5)
*   **Q: How does Platform Accuracy impact Business Value?**
    *   **A:** Accurate insights lead to higher trust, which increases retention (lower churn). This raises the CLV, allowing the company to spend more on acquiring high-value clients.
*   **Q: Which industry segment has the highest satisfaction?**
    *   **A:** In our simulation, **Agriculture** and **Disaster Management** show high satisfaction when accuracy is high, as the "cost of error" is very high in those fields.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Data Science:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Core:** Python 3.9

---
**GeoInsights Analytics Pvt. Ltd. © 2026**
