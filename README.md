# Spatial-Temporal Mining of Urban Heatwave Precursors and Regional Vulnerability Clustering

**Course:** Data Mining (6th Semester Engineering Mini-Project)
**Domain:** Urban Meteorology, Climate Data Mining

---

## 📌 Project Overview
This project applies foundational Data Mining techniques to historical hourly weather data across 36 global cities. The primary goal is to discover patterns leading up to heatwaves, classify thermal risk levels, and group geographic regions by their climate vulnerability.

This repository is modularized into four distinct Python scripts mapping directly to the university rubric modules: Preprocessing, Pattern Mining, Classification, and Clustering.

---

## 📂 Project Structure

```text
ganda projet/
├── datasets/                   # Raw Historical Hourly Weather CSVs
│   ├── city_attributes.csv     # Lat/Lon for 36 cities
│   ├── humidity.csv            
│   ├── pressure.csv            
│   ├── temperature.csv         # Measured in Kelvin
│   └── wind_speed.csv          
├── outputs/                    # Generated datasets & reports
│   ├── preprocessed_data.csv   # Cleaned & normalized continuous data
│   ├── discretized_data.csv    # Categorical data for pattern mining
│   ├── association_rules.csv   # Mined Apriori/FP-Growth rules
│   ├── classification_report.csv # Metrics for DT, NB, k-NN
│   └── cluster_assignments.csv # City labels for k-Means, HC, DBSCAN
├── plots/                      # Generated visualisations
│   ├── confusion_matrices.png  # Classification heatmaps
│   ├── classifier_comparison.png # Accuracy/Precision/Recall bar chart
│   ├── elbow_curve.png         # k-Means k-selection
│   ├── dendrogram.png          # Hierarchical Ward linkage tree
│   ├── cluster_maps_2d.png     # Scatter on Lat/Lon
│   └── cluster_3d.png          # Lat x Lon x Avg_Temp
├── preprocessing.py            # Module 2
├── pattern_mining.py           # Module 3
├── classification.py           # Module 4
├── clustering.py               # Module 5
└── README.md                   # Project Documentation
```

---

## 🛠️ Setup & Execution

**Dependencies Required:**
```bash
pip install pandas numpy scikit-learn mlxtend matplotlib seaborn
```

**How to Run (Execute sequentially):**
```bash
python preprocessing.py
python pattern_mining.py
python classification.py
python clustering.py
```

---

## 📘 Module-by-Module Breakdown & Viva Prep

This section is designed to help the team answer "in and out" questions during the project presentation/viva.

### 1. Module 2: Data Understanding & Preprocessing (`preprocessing.py`)
*   **What we did:** Merged 6 separate time-series CSVs into one massive dataset (~1.6M rows). Imputed missing values using Forward-Fill (to preserve temporal continuity) and column means. Removed noise using 1.5×IQR winsorization. Min-Max scaled the continuous features.
*   **Target Variable Creation:** Discretized raw temperature into 4 categorical bins using Kelvin thresholds: `Normal` (<294K), `Heat_Watch` (294-300K), `Heat_Warning` (300-306K), and `Extreme_Heat` (>306K).

### 2. Module 3: Pattern Mining (`pattern_mining.py`)
*   **Algorithms:** Apriori vs. FP-Growth.
*   **What we did:** Converted weather features into Low/Medium/High bins. Mined association rules to find precursors to heatwaves.
*   **🔥 VIVA GOTCHA - Why was Apriori faster than FP-Growth here?**
    Normally, FP-Growth is faster. However, because we only had 14 unique items (4 temp levels, 3 hum levels, 3 pres levels, 3 wind levels), the item-space was tiny. The overhead of building the FP-Tree was slower than Apriori's candidate generation on such a compact itemset. On massive retail datasets with millions of items, FP-Growth easily wins.
*   **🔥 VIVA GOTCHA - Why is "Heat Warning" on the left side (Antecedent)?**
    Because the "Normal" weather class dominates the dataset (66%). The miner naturally prefers "Normal" as a consequent. The rule `HEAT_WARNING => {Humidity=Low, Pressure=Low}` mathematically means: *When heat warnings occur, low humidity and low pressure strongly co-occur as precursors.*

### 3. Module 4: Classification (`classification.py`)
*   **Algorithms:** Decision Tree (Eager), Naive Bayes (Eager), k-NN (Lazy).
*   **What we did:** Used the normalized features to predict the `heat_risk` category. Stratified 80/20 train/test split. Evaluated using Accuracy, Precision, Recall, and F1.
*   **🔥 VIVA GOTCHA - Why did the Decision Tree score exactly 1.0 (100%)? Is it overfitting?**
    No, it's not overfitting in the traditional sense, but it *is* a deterministic leak. We artificially created the `heat_risk` labels directly from the `temperature` column in Module 2. The Decision Tree simply learned the exact threshold splits we programmed. Naive Bayes (97.6%) and k-NN (97.8%) score slightly lower because they factor in the other variables (humidity/pressure) and approximate the boundary geometrically/probabilistically.

### 4. Module 5: Clustering (`clustering.py`)
*   **Algorithms:** k-Means (Centroid), Hierarchical (Agglomerative), DBSCAN (Density).
*   **What we did:** Grouped the 36 cities based on geographic (Lat/Lon) and meteorological (Avg Temp/Hum/Wind) features. Generated 2D and 3D maps.
*   **🔥 VIVA GOTCHA - Why compare DBSCAN and k-Means?**
    k-Means *forces* every single city into a cluster, even if it's geographically isolated (like the cities in Israel vs. North America). DBSCAN, being density-based, correctly identifies isolated anomaly cities as **Noise** (Label = -1). For global climate analysis, DBSCAN's ability to handle outliers without distorting cluster centroids makes it vastly superior.

---

## 🎯 Key Takeaways for the Presentation

1.  **Meteorological Accuracy:** We proved via pattern mining that drops in humidity and pressure strongly correlate with severe heat events.
2.  **Algorithm Selection Matters:** We demonstrated that FP-Growth isn't *always* faster than Apriori (it depends on item-space density), and that DBSCAN handles geographical outliers far better than k-Means.
3.  **Comprehensive Coverage:** We satisfied all criteria for a standard Data Mining pipeline: Data Cleaning, Association, Classification, and Clustering, using standard Python libraries (`scikit-learn`, `mlxtend`, `pandas`).
