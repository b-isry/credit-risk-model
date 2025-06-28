import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str)-> pd.DataFrame:
    
    logging.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:   
        logging.error(f"File not found: {file_path}")
        raise

def create_customer_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level features from transaction data.
    """
    logging.info("Creating customer-level features")
    
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    logging.info("Creating aggregate transaction features")
    customer_agg = df.groupby('CustomerId').agg(
        total_transaction_amount = ('Value', 'sum'),
        avg_transaction_amount=('Value', 'mean'),
        std_transaction_amount=('Value', 'std'),
        transaction_count=('TransactionId', 'count'),
        unique_products_purchased=('ProductId', 'nunique'),
        unique_providers_used=('ProviderId', 'nunique')
    ).reset_index()
    
    customer_agg['std_transaction_amount'] = customer_agg['std_transaction_amount'].fillna(0)
    
    logging.info("Calculating RFM metrics")
    
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    # Calculate RFM values
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Value', 'sum')
    ).reset_index()

    # --- Merge Features ---
    logging.info("Merging aggregate features with RFM metrics...")
    customer_features = pd.merge(customer_agg, rfm, on='CustomerId')
    
    return customer_features

def create_proxy_target_variable(customer_features: pd.DataFrame) -> pd.DataFrame:
    """
    Uses K-Means clustering on RFM features to create a 'is_high_risk' proxy target.
    """
    logging.info("Creating proxy target variable using K-Means clustering...")
    
    # --- Select RFM features for clustering ---
    rfm_for_clustering = customer_features[['Recency', 'Frequency', 'Monetary']]
    
    # --- Pre-process for Clustering ---
    # Log transform Monetary and Frequency to handle skewness
    rfm_for_clustering['Monetary'] = np.log1p(rfm_for_clustering['Monetary'])
    rfm_for_clustering['Frequency'] = np.log1p(rfm_for_clustering['Frequency'])
    
    # Scale the features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_for_clustering)

    # --- Cluster Customers ---
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_features['rfm_cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # --- Define and Assign the "High-Risk" Label ---
    logging.info("Analyzing clusters to identify the high-risk segment...")
    cluster_summary = customer_features.groupby('rfm_cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    logging.info(f"\nCluster Summary (Mean Values):\n{cluster_summary}")
    
    # The high-risk cluster is typically the one with the highest Recency (least recent)
    # and lowest Frequency/Monetary.
    high_risk_cluster_id = cluster_summary['Recency'].idxmax()
    logging.info(f"Identified high-risk cluster ID: {high_risk_cluster_id}")
    
    # Create the binary target column
    customer_features['is_high_risk'] = np.where(customer_features['rfm_cluster'] == high_risk_cluster_id, 1, 0)
    
    # Drop the intermediate cluster column
    customer_features = customer_features.drop('rfm_cluster', axis=1)
    
    logging.info(f"High-risk proxy target created. Share of high-risk customers: {customer_features['is_high_risk'].mean():.2%}")
    
    return customer_features

def main():
    """
    Main function to run the data processing pipeline.
    """
    # Define file paths
    # Using relative paths from the root of the project
    raw_data_path = 'data/raw/data.csv'
    processed_data_path = 'data/processed/customer_features.csv'
    
    # Run the pipeline
    raw_df = load_data(raw_data_path)
    customer_features_df = create_customer_level_features(raw_df)
    final_df = create_proxy_target_variable(customer_features_df)
    
    # Save the final processed data
    logging.info(f"Saving processed data to {processed_data_path}...")
    final_df.to_csv(processed_data_path, index=False)
    logging.info("Data processing complete.")

if __name__ == '__main__':
    main()