# ☁️ Beijing Air Quality Analysis Dashboard

## 📌 Project Description
This project analyzes air quality conditions in Beijing using the PRSA Air Quality dataset. The analysis focuses on understanding pollution patterns, temporal trends, differences between monitoring stations, relationships with meteorological factors, and clustering air quality conditions using machine learning techniques.

The results are presented through:
- An exploratory data analysis notebook
- An interactive dashboard built with Streamlit

🔗 **Live Dashboard:**  
https://beijing-air-quality-analysis.streamlit.app/

---

## 📊 Dataset
The dataset used in this project is the **PRSA (Beijing Multi-Site Air Quality) Dataset**.

It contains air quality measurements collected from multiple monitoring stations in Beijing, including:

- PM2.5, PM10
- SO2, NO2, CO, O3
- Temperature (TEMP)
- Air Pressure (PRES)
- Wind Speed (WSPM)
- Time information (year, month, day, hour)
- Station names

Source: ```https://github.com/marceloreis/HTI/tree/master```

---

## 📁 Project Structure
```
beijing-air-quality-analysis/
│
├── notebooks/
│ └── ML_Mutia_Aulia.ipynb
│
├── dashboard/
│ ├── dashboard.py
│
├── data/
│ └── main_data_clean_final.csv
│
├── README.md
└── requirements.txt
```
---

## 🧰 Prerequisites

Before running this project, make sure you have:

### 1️⃣ Python Installed
Check Python version:
```bash
python --version
```
or

```bash
python3 --version
```
Recommended version: Python 3.8 or higher

If Python is not installed, download it from:
```https://www.python.org/downloads/```

⚠️ Make sure to check “Add Python to PATH” during installation.

### 2️⃣ pip Installed
Check pip:

```bash
pip --version
```
If pip is not available, install it using:

```bash
python -m ensurepip --upgrade
```
---

## ▶️ How to Run the Notebook and Dashboard Locally

### Step 1️⃣ Clone the Repository
```bash
git clone https://github.com/aomta/beijing-air-quality-analysis.git
```

```bash
cd beijing-air-quality-analysis
```

###  Step 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
🔧 Alternative: Install Libraries One by One
If the command above fails, install manually:

```bash
pip install streamlit folium pandas matplotlib seaborn scikit-learn scipy numpy jupyter ipykernel streamlit-folium plotly
```

---
## ▶️ Run Jupyter Notebook
```bash
jupyter notebook
```
Then open:
```
notebooks/ML_Mutia_Aulia.ipynb
```

---
## ▶️ Run the Dashboard (Locally)
### Step 1️⃣ Move to Dashboard Folder
```bash
cd dashboard
```
### Step 2️⃣ Run Streamlit App
``` bash
streamlit run dashboard.py
```

### Step 3️⃣ Open in Browser
```bash
http://localhost:8501
```

### 🚀 Deployment
The dashboard is deployed using Streamlit Cloud.
Entry file: ```dashboard/dashboard.py```


Live URL:
```https://beijing-air-quality-analysis.streamlit.app/```

---

## 🔍 Key Insights from the Analysis
Based on the Exploratory Data Analysis (EDA) and Machine Learning models applied to the Beijing Air Quality dataset (2013–2017), the following critical insights were observed:

### 1. Seasonal Pollution Patterns

- Winter Peaks: Air quality deteriorates significantly during the winter months (December, January, and February). This is attributed to the increased usage of coal-based heating systems and meteorological conditions like temperature inversion that trap pollutants.

- Summer Relief: The lowest PM2.5 concentrations are consistently recorded during summer (July and August), likely due to better atmospheric dispersion and higher precipitation.

### 2. Spatial Disparity (Urban vs. Suburban)
There is a noticeable air quality gap between stations.

- High Pollution Zones: Stations located in central and southern urban areas (e.g., Dongsi, Wanshouxigong) exhibit higher average pollutant concentrations due to heavy traffic and urban density.

- Cleaner Zones: Northern suburban stations (e.g., Huairou, Dingling) generally record better air quality, benefiting from their proximity to mountainous regions and lower population density.

### 3. Correlation with Weather Factors
- A strong negative correlation exists between Wind Speed (WSPM) and PM2.5 levels. Wind acts as a natural ventilation system; higher wind speeds effectively disperse particulate matter.

- Temperature also shows a negative correlation with pollution, reinforcing the observation that warmer months tend to have cleaner air.

### 4. Machine Learning Insights (Clustering)
Using K-Means Clustering, daily weather and pollution profiles were categorized into three distinct clusters:

- Cluster 0 (Clean Air): Characterized by high wind speeds. Under these conditions, PM2.5 levels remain low regardless of other factors.

- Cluster 1 (Transitional): Moderate wind speeds with fluctuating pollution levels.

- Cluster 2 (Hazardous/Stagnant): Characterized by near-zero wind speed (stagnant air). This condition leads to the drastic accumulation of pollutants, resulting in "Very Unhealthy" or "Hazardous" AQI levels.

### 5. Conclusion & Recommendation
The analysis confirms that stagnant air during the winter heating season poses the highest health risk. Future mitigation strategies should focus on stricter emission controls during low-wind forecast periods and enhancing green infrastructure in central urban zones.

---
## 📸 Dashboard Preview

![Dashboard Preview](https://raw.githubusercontent.com/aomta/beijing-air-quality-analysis/main/images/1.png)
![Dashboard Preview](https://raw.githubusercontent.com/aomta/beijing-air-quality-analysis/main/images/2.png)
![Dashboard Preview](https://raw.githubusercontent.com/aomta/beijing-air-quality-analysis/main/images/3.png)


---

👤 Author

Mutia Aulia

Undergraduate Information Systems Student 
Interest: Data Analysis, Machine Learning, and Data Visualization

---