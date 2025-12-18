# Automated Customer Segmentation Pipeline (Snowflake/Snowpark)

## Project Overview
This project is an unsupervised learning pipeline designed to ingest live consumer survey data and segment customers based on pandemic risk tolerance. It is built entirely within Snowflake using Snowpark Python to demonstrate a "Zero-Copy" data architecture.

## Tech Stack
* **Infrastructure:** Snowflake (Data Warehouse), Snowpark (Compute)
* **ML Library:** Scikit-Learn (K-Means Clustering)
* **Language:** Python 3.9
* **Data Source:** Snowflake Marketplace (Statista / Cybersyn)

## The Problem
Raw survey data containing 100+ columns is often too dense for business stakeholders to interpret quickly. Marketing teams required clear "Personas" rather than abstract statistics to tailor their messaging strategies effectively.

## The Solution
1.  **Ingestion:** Connected to live "Respondent Level Data" via Snowflake Marketplace.
2.  **Clustering:** Applied K-Means (k=3) to group respondents by their concern levels regarding health risks.
3.  **Translation Layer:** Automated the renaming of clusters from integers (0, 1, 2) to human-readable labels ("Risk-Averse", "Skeptics", "Cautious Observers").

## Results
* Identified 3 distinct behavioral tribes within the dataset.
* Eliminated ETL latency by executing ML workloads where the data resides.
* Produced a final "Persona Table" ready for consumption by BI tools (Tableau/PowerBI).

## How to Run


### Option 1: Snowflake Python Worksheet (Recommended)
This pipeline is designed to run natively within Snowflake's secure environment.

1.  **Data Setup:** Acquire the "Consumer Insights Respondent Level Data Sample" (Statista) from the Snowflake Marketplace (Free Tier).
2.  **Environment:** Open a new **Python Worksheet** in Snowsight.
3.  **Dependencies:** In the "Packages" dropdown, select:
    * `scikit-learn`
    * `pandas`
    * `snowflake-snowpark-python`
4.  **Deploy:** Copy the contents of `src/clustering_pipeline.py` into the worksheet and click **Run**.
5.  **Output:** The script will return a dataframe displaying the 3 generated customer personas.

### Option 2: Local Development (Advanced)
To run this locally, you must configure a `connection.json` file with your Snowflake credentials.

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Instantiate a local Snowpark session and call the `main()` function, passing the session object.
