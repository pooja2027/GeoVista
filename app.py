import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

st.title("Geo Clustering with KMeans")

uploaded_file = st.file_uploader("Upload CSV file with latitude and longitude columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # strip spaces
    
    # Debug print of columns
    st.write("Columns in your CSV:", df.columns.tolist())

    # Rename if needed (adjust based on your actual columns)
    df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

    if 'latitude' in df.columns and 'longitude' in df.columns:

        st.write("Raw Data")
        st.dataframe(df)

        imputer = SimpleImputer(strategy='mean')
        coords_imputed = imputer.fit_transform(df[['latitude', 'longitude']])

        n_clusters = st.slider("Select number of clusters", 2, 10, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(coords_imputed)

        st.write("Clustered Data")
        st.dataframe(df)

        centers = kmeans.cluster_centers_
        st.write("Cluster Centers (Latitude, Longitude)")
        st.write(pd.DataFrame(centers, columns=['latitude', 'longitude']))

    else:
        st.error("CSV must contain 'latitude' and 'longitude' columns.")

else:
    st.info("Please upload a CSV file to start clustering.")
