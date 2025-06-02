import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Title of your app
st.title("Geo Clustering with KMeans")

# Load your data
uploaded_file = st.file_uploader("Upload CSV file with latitude and longitude columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show raw data
    st.write("Raw Data")
    st.dataframe(df)

    # Check required columns
    if 'latitude' in df.columns and 'longitude' in df.columns:

        # Impute missing lat/lon with mean
        imputer = SimpleImputer(strategy='mean')
        coords_imputed = imputer.fit_transform(df[['latitude', 'longitude']])

        # Select number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(coords_imputed)

        # Assign clusters to dataframe
        df['cluster'] = clusters

        st.write("Clustered Data")
        st.dataframe(df)

        # Optional: Show cluster centers
        centers = kmeans.cluster_centers_
        st.write("Cluster Centers (Latitude, Longitude)")
        st.write(pd.DataFrame(centers, columns=['latitude', 'longitude']))

    else:
        st.error("CSV must contain 'latitude' and 'longitude' columns.")

else:
    st.info("Please upload a CSV file to start clustering.")
