import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Beijing Air Quality Comprehensive Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('dark_background')
sns.set_context("notebook")

# Koordinat Stasiun
STATION_COORDS = {
    "Aotizhongxin": [39.982, 116.397], "Changping": [40.217, 116.230],
    "Dingling": [40.292, 116.220], "Dongsi": [39.929, 116.417],
    "Guanyuan": [39.929, 116.339], "Gucheng": [39.914, 116.184],
    "Huairou": [40.328, 116.628], "Nongzhanguan": [39.937, 116.461],
    "Shunyi": [40.127, 116.655], "Tiantan": [39.886, 116.407],
    "Wanliu": [39.987, 116.287], "Wanshouxigong": [39.878, 116.352]
}

# Fungsi sederhana hitung AQI (US EPA Standard Approximation)
def calculate_aqi(concentration, pollutant):
    # Breakpoints sederhana
    breakpoints = {
        'PM2.5': [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
        'PM10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
        'SO2': [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
        'NO2': [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200)],
        'CO': [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)],
        'O3': [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200)]
    }
    
    if pollutant not in breakpoints:
        return 0
    
    c = concentration
    for (clo, chi, ilo, ihi) in breakpoints[pollutant]:
        if clo <= c <= chi:
            return ((ihi - ilo) / (chi - clo)) * (c - clo) + ilo
    
    # Jika melebihi batas atas, anggap hazardous max
    if c > breakpoints[pollutant][-1][1]:
        return 500
    return 0

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../data/main_data_clean_final.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['PM2.5'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['hour'] = df['datetime'].dt.hour
        
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty:
    st.error("❌ The file 'main_data_clean_final.csv' was not found.")
    st.stop()

# SIDEBAR (FILTER)
with st.sidebar:
    st.title("Beijing AQI Dashboard")
    
    with st.expander("ℹ️ About the Dashboard", expanded=False):
        st.markdown("""
        **Project Description:**
        This dashboard presents a comprehensive analysis of air quality at 12 major stations in Beijing (2013-2017).
        
        **Key Features:**
        - 📈 **Trends:** Daily and seasonal pollution patterns.
        - 🗺️ **Map:** Spatial distribution of pollution.
        - 🤖 **ML:** Weather and pollution clustering.
        - 🌦️ **Correlation:** Relationship between temperature, wind, etc.
        
        **Data Source:** PRSA Data Set.
        **Developed by:** aomta | 2025
        """)
        
    st.header("🎛️ Filter Dashboard")
    min_date = df_raw['datetime'].min().date()
    max_date = df_raw['datetime'].max().date()
    date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    all_stations = sorted(df_raw['station'].unique())
    select_all = st.checkbox("Select All Stations", value=True)
    if select_all:
        selected_stations = all_stations
    else:
        selected_stations = st.multiselect("Select Stations:", all_stations, default=all_stations[:1])
    
if len(date_range) == 2:
    start, end = date_range
    main_df = df_raw[(df_raw['datetime'].dt.date >= start) & (df_raw['datetime'].dt.date <= end) & (df_raw['station'].isin(selected_stations))].copy()
else:
    main_df = df_raw[df_raw['station'].isin(selected_stations)].copy()

if main_df.empty:
    st.warning("No data with this filter.")
    st.stop()

# 4. DASHBOARD UTAMA
st.title("🌤️ Beijing Air Quality Analysis")
st.markdown(f"**Data:** {len(main_df):,} rows | **Stasiun:** {len(selected_stations)}")

col1, col2, col3, col4 = st.columns(4)
avg_pm25 = main_df['PM2.5'].mean()
max_pm25 = main_df['PM2.5'].max()
avg_temp = main_df['TEMP'].mean()
avg_wind = main_df['WSPM'].mean()

col1.metric("Average PM2.5", f"{avg_pm25:.2f} µg/m³", delta_color="inverse")
col2.metric("Maximum PM2.5", f"{max_pm25:.0f} µg/m³", "Danger" if max_pm25 > 150 else "Safe")
col3.metric("Average Temperature", f"{avg_temp:.1f} °C")
col4.metric("Average Wind Speed", f"{avg_wind:.2f} m/s")

# --- TABS ---
# TABS DEFINITION
tab_summary, tab_trend, tab_comp, tab_dist, tab_heat, tab_corr, tab_map, tab_ml = st.tabs([
    "📋 Data Summary", "📈 Time Trend", "📊 Comparison", "🎯 AQI Distribution", 
    "🔥 Heatmap", "☁️ Correlation", "🌍 Spatial Map", "🤖 Machine Learning"
])

# TAB 1: RINGKASAN DATA (DESCRIPTIVE STATISTICS)
with tab_summary:
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    st.subheader("Summary of statistical data (descriptive statistics)")
    
    col_sum1, col_sum2 = st.columns([3, 1])
    
    with col_sum1:
        st.markdown("**Descriptive Statistics (Mean, Median, Std, Min, Max)**")
        numeric_cols = pollutants + ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        
        # Buat Describe dataframe dan Transpose agar mudah dibaca
        desc_df = main_df[numeric_cols].describe().T
        
        # Formatting tampilan agar tidak terlalu banyak desimal
        st.dataframe(desc_df.style.format("{:.2f}"), use_container_width=True, height=400)
        
    with col_sum2:
        st.markdown("**Dataset Info**")
        st.metric("Total number of data rows", f"{len(main_df):,}")
        st.metric("Number of Stations", f"{len(selected_stations)}")
        st.metric("Number of Pollutants", f"{len(pollutants)}")
        st.info("The table on the left shows a summary of statistical data from the filtered dataset.")

    st.markdown("---")
    st.markdown("**3. Raw Data Preview**")
    st.dataframe(main_df.head(), use_container_width=True)

# TAB 2: TREN WAKTU (Konsentrasi & AQI)
with tab_trend:
    st.subheader("Pollution Trends & AQI Scores")
    
    col_p, col_mode = st.columns([1, 3])
    with col_p:
        sel_pol_trend = st.selectbox("Select Pollutant:", pollutants, index=0)
    
    daily_data = main_df.set_index('datetime').resample('W')[sel_pol_trend].mean().reset_index()
    daily_data['AQI_Est'] = daily_data[sel_pol_trend].apply(lambda x: calculate_aqi(x, sel_pol_trend))
    
    # Plot 1: Tren Konsentrasi
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(daily_data['datetime'], daily_data[sel_pol_trend], color='cyan', linewidth=1.5)
    ax1.set_title(f"Average Yearly Trend: {sel_pol_trend} (Concentration)", color='white')
    ax1.set_ylabel("µg/m³", color='white')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(colors='white')
    st.pyplot(fig1)
    
    # Plot 2: Tren AQI
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(daily_data['datetime'], daily_data['AQI_Est'], color='magenta', linewidth=1.5, linestyle='-')
    ax2.set_title(f"Average Monthly AQI Score Trend ({sel_pol_trend})", color='white')
    ax2.set_ylabel("AQI Score", color='white')
    ax2.axhline(100, color='yellow', linestyle='--', label='Unhealthy Limit')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(colors='white')
    st.pyplot(fig2)

# TAB 2: PERBANDINGAN STASIUN (Comparison)
with tab_comp:
    st.subheader("Comparison Between Stations")
    sel_pol_comp = st.selectbox("Select Pollutant for Comparison:", pollutants, key='comp')
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        # Plot 1: Average Physical Concentration per Station
        st.markdown(f"**1. Average Concentration {sel_pol_comp}**")
        avg_conc = main_df.groupby('station')[sel_pol_comp].mean().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.barplot(x=avg_conc.values, y=avg_conc.index, palette='viridis', ax=ax3)
        ax3.set_xlabel("Concentration (µg/m³)")
        ax3.tick_params(colors='white')
        st.pyplot(fig3)
        
    with col_c2:
        # Plot 2: Average AQI Score per Station
        st.markdown(f"**2. Average AQI Score ({sel_pol_comp})**")
        # Kita hitung AQI rata-rata kasar dari rata-rata konsentrasi (approximation)
        avg_aqi_st = avg_conc.apply(lambda x: calculate_aqi(x, sel_pol_comp)).sort_values(ascending=False)
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.barplot(x=avg_aqi_st.values, y=avg_aqi_st.index, palette='magma', ax=ax4)
        ax4.set_xlabel("AQI Score")
        ax4.tick_params(colors='white')
        st.pyplot(fig4)
    
    # Plot 3: Pollutant Concentration Profile (Hourly)
    st.markdown("---")
    st.markdown(f"**3. Hourly Profile {sel_pol_comp} per Station (Average per Hour)**")
    hourly_profile = main_df.groupby(['hour', 'station'])[sel_pol_comp].mean().reset_index()
    
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=hourly_profile, x='hour', y=sel_pol_comp, hue='station', palette='tab10', ax=ax5, linewidth=1)
    ax5.set_title(f"Hourly Concentration Pattern {sel_pol_comp}", color='white')
    ax5.set_xlabel("Hour (0-23)")
    ax5.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax5.grid(True, linestyle='--', alpha=0.3)
    ax5.tick_params(colors='white')
    st.pyplot(fig5)

# TAB 3: DISTRIBUSI KATEGORI AQI
with tab_dist:
    st.subheader("Distribution of Air Quality Categories")
    
    # Hitung AQI untuk polutan yang dipilih user (Sample 5000 agar cepat)
    # Note: Menghitung AQI untuk semua data (400k baris) bisa lama, kita gunakan sampling untuk pie chart distribusi
    dist_sample = main_df.sample(min(10000, len(main_df))).copy()
    
    sel_pol_dist = st.selectbox("Select Pollutant:", pollutants, key='dist')
    dist_sample['AQI_Temp'] = dist_sample[sel_pol_dist].apply(lambda x: calculate_aqi(x, sel_pol_dist))
    dist_sample['Category'] = dist_sample['AQI_Temp'].apply(get_aqi_category)
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown(f"**AQI Category Distribution ({sel_pol_dist})**")
        cat_counts = dist_sample['Category'].value_counts()
        colors_aqi = {'Good':'#00e400', 'Moderate':'#ffff00', 'Unhealthy for Sensitive Groups':'#ff7e00', 
                      'Unhealthy':'#ff0000', 'Very Unhealthy':'#8f3f97', 'Hazardous':'#7e0023'}
        
        fig6, ax6 = plt.subplots()
        # Urutkan label agar warna konsisten
        labels = [l for l in cat_counts.index]
        colors = [colors_aqi.get(l, 'gray') for l in labels]
        
        ax6.pie(cat_counts, labels=labels, autopct='%1.1f%%', colors=colors, textprops={'color':"white"})
        st.pyplot(fig6)
        
    with col_d2:
        st.markdown("**Distribution of AQI Categories per Pollutant (Bar Chart)**")
        data_stack = []
        for p in pollutants:
            val = dist_sample[p].mean()
            aqi_val = calculate_aqi(val, p)
            cat = get_aqi_category(aqi_val)
            data_stack.append({'Pollutant': p, 'Avg_AQI': aqi_val, 'Category': cat})
        
        df_stack = pd.DataFrame(data_stack)
        fig7, ax7 = plt.subplots()
        sns.barplot(data=df_stack, x='Pollutant', y='Avg_AQI', hue='Category', dodge=False, ax=ax7, palette='Reds')
        ax7.set_title("Average AQI per Pollutant", color='white')
        ax7.tick_params(colors='white')
        st.pyplot(fig7)

# TAB 4: HEATMAP TEMPORAL (Month vs Hour)
with tab_heat:
    st.subheader("Heatmap Temporal: Month vs Hour")
    st.caption("Viewing pollution patterns based on time within a year.")

    sel_pol_heat = st.selectbox("Select Pollutant:", pollutants, key='heat')

    # Pivot Table: Index=Month, Columns=Hour
    heatmap_data = main_df.pivot_table(index='month', columns='hour', values=sel_pol_heat, aggfunc='mean')
    
    fig8, ax8 = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=False, fmt=".1f", ax=ax8)
    ax8.set_title(f"Concentration {sel_pol_heat} (Month vs Hour)", color='white')
    ax8.set_xlabel("Hour (0-23)", color='white')
    ax8.set_ylabel("Month (1-12)", color='white')
    ax8.tick_params(colors='white')
    st.pyplot(fig8)

# TAB 5: KORELASI CUACA
with tab_corr:
    st.subheader("Relationship Between Weather Factors and Pollution")
    
    # Scatter Plot Matrix or Heatmap
    weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    target_cols = pollutants
    
    corr_df = main_df[weather_cols + target_cols].corr()
    
    fig9, ax9 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, cmap='RdBu_r', annot=True, fmt=".2f", center=0, ax=ax9)
    ax9.set_title("Correlation Matrix", color='white')
    ax9.tick_params(colors='white')
    st.pyplot(fig9)

    st.info("Negative Correlation (Dark Red/Blue) indicates an inverse relationship. Example: WSPM (Wind) increases, PM2.5 decreases.")

# TAB 6: PETA SPASIAL
with tab_map:
    st.subheader("Map of Air Quality Distribution (Average Period)")
    st.caption("🟢 Good | 🟡 Moderate | 🟠 Unhealthy for Sensitive Groups | 🔴 Unhealthy | 🟣 Hazardous")
    
    # Aggregasi data per stasiun
    map_agg = main_df.groupby('station')['PM2.5'].mean().reset_index()
    
    # Inisialisasi Peta
    m = folium.Map(location=[40.0, 116.4], zoom_start=10, tiles='CartoDB dark_matter')
    
    for idx, row in map_agg.iterrows():
        s_name = row['station']
        pm_val = row['PM2.5']
        
        if pm_val <= 35:
            icon_color = 'green'
            hex_color = '#00FF00' 
            status = "Good (Baik)"
        elif pm_val <= 75:
            icon_color = 'beige'   
            hex_color = '#FFFF00'
            status = " Moderate(Sedang)"
        elif pm_val <= 115:
            icon_color = 'orange'
            hex_color = '#FFA500' 
            status = "Unhealthy for Sensitive Groups"
        elif pm_val <= 150:
            icon_color = 'red'
            hex_color = '#FF0000' 
            status = "Unhealthy (Tidak Sehat)"
        else:
            icon_color = 'purple'
            hex_color = '#800080' 
            status = "Hazardous (Berbahaya)"
        
        if s_name in STATION_COORDS:
            lat, lon = STATION_COORDS[s_name]
            
            folium.Marker(
                [lat, lon],
                popup=f"""
                <div style='font-family: Arial; font-size: 12px;'>
                    <b>{s_name}</b><br>
                    PM2.5: <b>{pm_val:.2f}</b><br>
                    Status: <span style='color:{hex_color};'><b>{status}</b></span>
                </div>
                """,
                tooltip=f"{s_name}: {status}",
                icon=folium.Icon(color=icon_color, icon='cloud', prefix='fa')
            ).add_to(m)
            
            folium.Circle(
                [lat, lon],
                radius=2500,
                color=hex_color, 
                fill=True,
                fill_color=hex_color,
                fill_opacity=0.5
            ).add_to(m)
            
    st_folium(m, width=1000, height=500)

#TAB 7: MACHINE LEARNING (K-Means)
with tab_ml:
    st.subheader("🤖 Clustering of pollution patterns (K-means)")
    
    col_ml1, col_ml2 = st.columns([1, 3])
    
    with col_ml1:
        st.markdown("**Model Settings**")
        n_clusters = 3
        sample_size = st.slider("Number of Data Samples:", 500, 8000, 2000, step=500)
        st.caption(f"Fixed clusters: {n_clusters}")
    
    with col_ml2:
        # Sampling
        if len(main_df) > sample_size:
            ml_data = main_df.sample(sample_size, random_state=42).copy()
        else:
            ml_data = main_df.copy()
            
        features = ['PM2.5', 'WSPM']
        X = ml_data[features].dropna()
        
        if len(X) > 50:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            ml_data['Cluster'] = clusters
            
            means = ml_data.groupby('Cluster')['PM2.5'].mean().sort_values()
            labels_map = {
                means.index[0]: 'Good (High Wind)',
                means.index[1]: 'Moderate',
                means.index[2]: 'Polluted (Stagnant)'
            }
            ml_data['Label'] = ml_data['Cluster'].map(labels_map)
            cluster_colors = {'Good (High Wind)': '#2ca02c', 'Moderate': '#ff7f0e', 'Polluted (Stagnant)': "#f11f1f"}
                    
            # Scatter Plot
            fig_ml, ax_ml = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=ml_data, x='WSPM', y='PM2.5', hue='Label', palette=cluster_colors, s=50, ax=ax_ml)
            ax_ml.set_title("Clustering: Wind vs PM2.5", color='white')
            ax_ml.set_xlabel("Wind Speed (m/s)", color='white')
            ax_ml.set_ylabel("PM2.5", color='white')
            
            handles, labels = ax_ml.get_legend_handles_labels()
            ordered_labels = list(labels_map.values())
            ordered_handles = [handles[labels.index(label)] for label in ordered_labels]
            
            ax_ml.legend(handles=ordered_handles, labels=ordered_labels, labelcolor='white', facecolor='black', edgecolor='white', title="Type of Day Condition")
            ax_ml.tick_params(colors='white')
            st.pyplot(fig_ml)
        else:
            st.warning("There is insufficient data.")

# Footer
st.markdown("---")
st.caption("Beijing Air Quality Dashboard | Streamlit Integration | 2025 | developed by aomta")