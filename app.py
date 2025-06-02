import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🌍 GeoVista: Geospatial Market Insights for AEC Industry")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your AEC project CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Strip spaces

    # Rename common variants
    df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

    st.subheader("📈 Raw Data")
    st.dataframe(df.head())

    # Check required columns
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Remove rows with NaNs in lat/lon
        df = df.dropna(subset=['latitude', 'longitude'])

        # Clustering
        st.sidebar.header("🔧 Clustering Parameters")
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

        imputer = SimpleImputer(strategy='mean')
        coords = imputer.fit_transform(df[['latitude', 'longitude']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(coords)

        # Show Clustered Data
        st.subheader("📊 Clustered Data")
        st.dataframe(df)

        # Show Chart
        st.subheader("📌 Cluster Count")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='cluster', palette='viridis')
        plt.title("Projects per Cluster")
        st.pyplot(fig)

        # Show Folium Map
        st.subheader("🗺️ Cluster Map")
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'black']

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                color=colors[row['cluster'] % len(colors)],
                fill=True,
                fill_opacity=0.7,
                popup=str(row.get('project_name', f"Cluster {row['cluster']}"))
            ).add_to(m)

        st_folium(m, width=700, height=500)

    else:
        st.error("CSV must contain 'latitude' and 'longitude' columns.")
else:
    st.info("Upload a CSV file to begin.")
