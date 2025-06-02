import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# Title
st.set_page_config(page_title="GeoVista Dashboard", layout="wide")
st.title("🌍 GeoVista: Geospatial Market Insights for AEC Industry")
st.markdown("Upload your AEC project CSV file")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()  # make column names lowercase

    st.subheader("📈 Raw Data")
    st.dataframe(df.head())

    # Check for required columns
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Select number of clusters
        k = st.slider("🔢 Select number of clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, n_init='auto')
        df['cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

        # Show Cluster Map
        st.subheader("🗺️ Cluster Map")
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)

        cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                          'lightblue', 'pink', 'gray', 'cadetblue']

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                color=cluster_colors[row['cluster'] % len(cluster_colors)],
                fill=True,
                fill_opacity=0.7,
                popup=row.get('project_name', 'Project')
            ).add_to(m)

        st_folium(m, width=800, height=500)
    else:
        st.error("❌ CSV must contain columns named 'latitude' and 'longitude'. Please check your file.")
else:
    st.info("📌 Please upload a CSV file to begin.")
