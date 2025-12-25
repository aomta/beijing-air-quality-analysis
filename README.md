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
* PM2.5 concentrations vary significantly across different monitoring stations.

* Monthly and seasonal trends indicate periods of higher pollution levels.

* Meteorological factors such as temperature and wind speed show relationships with PM2.5 concentration.

* Clustering using K-Means successfully groups air quality conditions into distinct categories representing low, medium, and high pollution levels.

* The interactive dashboard enables users to explore air quality data dynamically across time and stations.

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