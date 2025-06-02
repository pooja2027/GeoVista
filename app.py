import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
import altair as alt

# App title
st.set_page_config(page_title="Geo Clustering with KMeans", layout="wide")
st.title("🌍 Geo Clustering with KMeans")
st.markdown("Upload a CSV containing **latitude** and **longitude** columns to perform spatial clustering and view insightful analytics.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

# Processing logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

    # Rename for consistency
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    df.rename(columns={'latitude': 'latitude', 'longitude': 'longitude'}, inplace=True)

    # Column check
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.success("✅ Latitude and Longitude columns found!")
        st.subheader("📄 Raw Data")
        st.dataframe(df)

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        coords_imputed = imputer.fit_transform(df[['latitude', 'longitude']])

        # KMeans clustering
        n_clusters = st.slider("🔢 Select number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(coords_imputed)

        st.subheader("🧠 Clustered Data")
        st.dataframe(df)

        # Display cluster centers
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
        st.subheader("📍 Cluster Centers")
        st.dataframe(centers)

        # Show map
        st.subheader("🗺️ Cluster Map")
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
        cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cadetblue', 'gray', 'beige']

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                color=cluster_colors[int(row['cluster']) % len(cluster_colors)],
                fill=True,
                popup=str(row.get('project_name', f"Cluster {row['cluster']}"))
            ).add_to(m)

        st_folium(m, width=900, height=500)

        # Analytics Section
        with st.expander("📊 Show Charts & Insights"):
            if 'project_type' in df.columns:
                # Project count
                count_chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('project_type:N', title='Project Type'),
                    y=alt.Y('count():Q', title='Number of Projects'),
                    color='project_type:N'
                ).properties(title='Project Count by Type', width=600)
                st.altair_chart(count_chart)

                # Average cost
                if 'cost' in df.columns:
                    avg_cost = df.groupby('project_type')['cost'].mean().reset_index()
                    st.subheader("💰 Average Project Cost by Type")
                    st.bar_chart(avg_cost.rename(columns={'project_type': 'index'}).set_index('index'))

            # Project completion trend
            if 'completion_year' in df.columns:
                trend = df['completion_year'].value_counts().sort_index().reset_index()
                trend.columns = ['Year', 'Number of Projects']
                st.subheader("📆 Projects Completed Over Time")
                st.line_chart(trend.set_index('Year'))

            # Cost distribution
            if 'cost' in df.columns:
                st.subheader("📈 Cost Distribution")
                st.histogram(df['cost'], bins=10)

    else:
        st.error("❌ CSV must contain 'latitude' and 'longitude' columns.")

else:
    st.info("📂 Please upload a CSV file to start clustering.")
