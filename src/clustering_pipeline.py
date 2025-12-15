import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
from sklearn.cluster import KMeans

def main(session: snowpark.Session):
    """
    Orchestrates the K-Means clustering pipeline.
    1. Connects to Snowflake Shared Data.
    2. Performs data extraction.
    3. Executes K-Means clustering via Scikit-Learn.
    4. Translates clusters to business personas.
    """
    
    # 1. SWITCH CONTEXT: Move to writable schema for temp tables
    # Replace 'YOUR_DB' with your actual database name if running locally
    session.use_database("MISTERPLOW_DATA") 
    session.use_schema("PUBLIC")
    
    # 2. LOAD DATA
    # Pointing to the Marketplace Data (Statista/Cybersyn)
    # Note: In a prod env, this would be a variable or config
    table_name = "CONSUMER_DB.CONSUMER_INSIGHTS_SCHEMA.RLD_SAMPLE"
    
    # Select feature columns (Coronavirus Concern Levels)
    df = session.table(table_name).select(
        col("YEAR"), 
        col("CORONAVIRUS"), 
        col('"CoronaVirus2#1"'), 
        col('"CoronaVirus2#2"')
    )

    # 3. PRE-PROCESSING (Pandas)
    # Convert to local Pandas DF for Scikit-Learn processing
    pdf = df.to_pandas()
    pdf = pdf.dropna()
    
    # 4. MODELING (K-Means)
    # Segmenting into 3 distinct "Tribes"
    kmeans = KMeans(n_clusters=3, random_state=42)
    
    features = pdf[['CORONAVIRUS', 'CoronaVirus2#1', 'CoronaVirus2#2']]
    pdf['CLUSTER_LABEL'] = kmeans.fit_predict(features)
    
    # 5. BUSINESS LOGIC LAYER
    # Summarize centroids to interpret the segments
    summary = pdf.groupby('CLUSTER_LABEL').mean().reset_index()

    def name_that_segment(row):
        """Maps numerical centroids to human-readable personas."""
        if row['CORONAVIRUS'] > 3.5:
            return "High Anxiety / Risk Averse"
        elif row['CORONAVIRUS'] < 2.5:
            return "Skeptics / Business-as-Usual"
        else:
            return "Cautious Observers"

    # Apply translation logic
    summary['PERSONA_NAME'] = summary.apply(name_that_segment, axis=1)
    
    # Clean up for final report
    final_report = summary[['PERSONA_NAME', 'CORONAVIRUS', 'CoronaVirus2#1']]
    
    return session.create_dataframe(final_report)
