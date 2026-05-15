# MODULE 1: Problem Identification & Domain Understanding
## Course: Data Mining Mini Project (6th Semester)
### Domain: Urban Meteorology / Climate Data Mining
### Dataset: Historical Hourly Weather Data across 36 Global Cities

---

## 1. Problem Statement

Extreme heat events and urban heatwaves are becoming increasingly frequent and severe due to global climate change and the **Urban Heat Island (UHI) effect**. These thermal anomalies pose severe threats to public health, strain electrical grids, and disrupt urban infrastructure. 

**The core problem is:** Given a massive stream of historical, hourly meteorological data across global cities — described by spatial, temporal, and continuous weather attributes — can we automatically discover the meteorological precursors to heatwaves, classify thermal risk levels, and group geographic regions by their climate vulnerability, while minimizing:

- **False Negatives (Missed Warnings):** Failing to predict or classify a severe heatwave leads to a lack of public health advisories, resulting in increased heat-related morbidity and unprepared power grids.
- **False Positives (False Alarms):** Incorrectly predicting extreme heat leads to unnecessary municipal resource deployment, school/business closures, and eventual "warning fatigue" among the public.

This is fundamentally a **pattern discovery, classification, and spatial clustering problem**, where understanding the multivariate co-occurrence of weather variables is critical for proactive urban climate resilience.

---

## 2. Types of Data in Climate & Urban Meteorology Datasets

The dataset used in this project (~1.6 million hourly records merged from 6 distinct sources) contains four primary categories of attributes:

### 2.1 Spatial / Geographic Data
| Attribute | Type | Example Values | Role in Vulnerability Clustering |
|-----------|------|----------------|------------------------|
| `city` | Nominal (Categorical) | Vancouver, Miami, Tel Aviv | Serves as the primary entity for aggregation and geographic profiling |
| `Country` | Nominal (Categorical) | United States, Israel | Macro-level geographic grouping |
| `Latitude` | Numeric (Continuous) | 37.7749, 49.2496 | Core spatial feature for determining geographic climate zones |
| `Longitude` | Numeric (Continuous) | -122.4194, 34.7913 | Core spatial feature for clustering |

### 2.2 Meteorological / Continuous Data
| Attribute | Type | Example Values | Role in Pattern Mining |
|-----------|------|----------------|------------------------|
| `temperature` | Numeric (Ratio, Kelvin) | 283.9 K, 308.1 K | The primary indicator of heat; used to derive the target class |
| `humidity` | Numeric (Ratio, %) | 31.6%, 81.9% | Crucial co-factor in heat index (perceived heat) and a strong precursor variable |
| `pressure` | Numeric (Continuous) | 997.7, 1021.2 | Drops or spikes in atmospheric pressure signal shifting weather fronts |
| `wind_speed` | Numeric (Ratio) | 1.2 m/s, 3.7 m/s | Low wind speeds exacerbate the Urban Heat Island effect by trapping stagnant air |

### 2.3 Derived Categorical / Discretized Data
| Attribute | Type | Example Values | Role in Pattern Mining |
|-----------|------|----------------|------------------------|
| `heat_risk` (Target) | Ordinal (Categorical) | Normal, Heat_Watch, Heat_Warning, Extreme_Heat | Discretized from raw temperature to serve as the target class for prediction and rule mining |
| `hum_level` | Ordinal | Low, Medium, High | Discretized tertiles required for Apriori/FP-Growth |
| `pres_level` | Ordinal | Low, Medium, High | Discretized tertiles required for Apriori/FP-Growth |
| `wind_level` | Ordinal | Low, Medium, High | Discretized tertiles required for Apriori/FP-Growth |

### 2.4 Temporal Data
| Attribute | Type | Example Values | Role in Pattern Mining |
|-----------|------|----------------|------------------------|
| `datetime` | Timestamp | 2012-10-01 13:00:00 | Establishes the temporal sequence; allows for forward-fill imputation to maintain time-series continuity |

---

## 3. Relevance of Data Mining in Urban Meteorology

### 3.1 Why Traditional Forecasting Methods Need Data Mining
Traditional Numerical Weather Prediction (NWP) relies heavily on complex, computationally expensive physics/fluid-dynamics equations to forecast short-term weather. However, they face challenges in:
- **Discovering long-term statistical associations:** Identifying hidden co-occurrences (e.g., specific humidity drops leading to extreme heat days later).
- **Spatial anomaly detection:** Grouping cities by multi-dimensional vulnerability rather than just physical proximity.
- **Data scale:** Processing millions of historical records to find generalized, actionable rules.

### 3.2 How Data Mining Addresses These Challenges
Data mining techniques provide a **statistical, pattern-driven** approach to supplement physics-based models:

| Technique | Application in Urban Meteorology |
|-----------|-------------------------------|
| **Association Rule Mining** (Apriori, FP-Growth) | Discovering meteorological precursors (e.g., {humidity=Low, pressure=Low} → heat=Heat_Warning) |
| **Classification** (Decision Tree, Naïve Bayes, k-NN) | Supervised learning to categorize the thermal risk profile of a given timestamp |
| **Clustering** (k-Means, Hierarchical, DBSCAN) | Unsupervised learning to group cities into vulnerability zones based on latitude, longitude, and average climate |
| **Anomaly Detection** | Using density-based clustering (DBSCAN) to identify geographically isolated cities (Noise = -1) with unique, anomalous climate profiles |

### 3.3 The Interdisciplinary Connection
This project sits at the intersection of:
- **Data Science / ML:** Feature scaling (Min-Max), noise removal (Winsorization), and algorithmic evaluation.
- **Meteorology & Climatology:** Understanding the physical relationships between pressure, humidity, and extreme thermal events.
- **Urban Planning:** Utilizing vulnerability clusters to dictate where cities need better cooling infrastructure or heat-action plans.

---

## 4. Expected Outcomes

By the end of this 5-module project, we expect to deliver:

1. **A robust preprocessing pipeline** that merges multiple time-series, handles missing values temporally (forward-fill), caps noise using 1.5×IQR Winsorization, scales features, and discretizes targets for mining.

2. **Actionable association rules** mined using Apriori and FP-Growth, proving that drops in specific atmospheric conditions act as leading indicators for heat warnings, complete with Support, Confidence, and Lift metrics.

3. **A comparative classification study** evaluating eager (Decision Tree, Naïve Bayes) vs. lazy (k-NN) learners to predict the `heat_risk` category, evaluated via Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

4. **Regional vulnerability mapping** utilizing k-Means, Agglomerative Hierarchical (Ward Linkage), and DBSCAN clustering. A key outcome will be demonstrating DBSCAN's superiority in handling geographic outliers compared to k-Means.

5. **Visual analytics** including 2D geographic scatter maps, 3D thermal vulnerability plots, Hierarchical dendrograms, and Elbow curves for k-selection.

---

## 5. Recent Trends in Climate Data Mining (2024–2025)

| Trend | Description | Relevance to This Project |
|-------|-------------|--------------------------|
| **AI-Driven Weather Models** | Models like Google GraphCast are replacing traditional physics engines using pure data-mining and deep learning | Highlights the shift towards historical data-driven prediction over pure numerical physics |
| **Density-Based Spatial Anomalies** | Using advanced clustering to find localized climate anomalies (micro-climates) | We directly implement DBSCAN to identify cities that act as climatic outliers (Noise) |
| **Explainable Climate AI (XAI)** | Policymakers require transparent models to allocate infrastructure budgets safely | Our use of Decision Trees and Association Rules provides perfect, transparent "If-Then" logic for planners |
| **Urban Heat Island (UHI) Mapping** | Granular data mining to find specific urban pockets that trap heat | Our 5-dimensional clustering (Lat, Lon, Temp, Hum, Wind) models macro-level UHI effects |

### Key Statistics:
- Extreme heat is the **deadliest weather-related hazard**, causing more fatalities annually than floods, hurricanes, and tornadoes combined.
- AI and Data Mining models have recently demonstrated the ability to predict severe weather anomalies up to **10 days in advance** with unprecedented accuracy.
- By 2050, over **68% of the world population** will live in urban areas, making localized thermal vulnerability clustering critical for future city planning.

---

## 6. Dataset Summary

| Property | Value |
|----------|-------|
| **Domain** | Urban Meteorology |
| **Total Hourly Records** | ~1.6 Million (Pre-cleaning) |
| **Unique Cities** | 36 Global Metropolitan Areas |
| **Target Variable** | `heat_risk` (4 Discretized Classes: Normal, Watch, Warning, Extreme) |
| **Numerical Features** | Temperature (K), Humidity (%), Pressure (hPa), Wind Speed (m/s) |
| **Geographic Features** | Latitude, Longitude |
| **Data Imputation** | Forward-fill (Temporal), Column Mean |
| **Scaling & Noise** | Min-Max Normalization, 1.5×IQR Winsorization |

---

*Prepared for BCI606 Data Mining Mini Project — Module 1*