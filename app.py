import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# Title
st.title("🌍 GeoVista: Geospatial Market Insights for AEC Industry")

# File uploader
uploaded_file = st.file_uploader("Upload your AEC project CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📈 Raw Data")
    st.dataframe(df.head())

    # Clustering - requires 'latitude' and 'longitude' columns
    if 'latitude' in df.columns and 'longitude' in df.columns:
        k = st.slider("Select number of clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k)
        df['cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
        
        # Plot map
        st.subheader("🗺️ Cluster Map")
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=f"#{(row['cluster']+1)*3:02x}00{(10-row['cluster'])*3:02x}",
                fill=True,
                popup=row.get('project_name', 'Project'),
            ).add_to(m)
        st_data = st_folium(m, width=700, height=500)
    else:
        st.error("CSV must contain 'latitude' and 'longitude' columns.")
else:
    st.info("Please upload a CSV file to proceed.")
