# ⚡ End-to-End Data Pipeline with Databricks, PySpark & dbt
<br>

### 📌 Project Description

This project demonstrates building a modern data pipeline using the **Databricks Lakehouse Platform**. It follows the medallion architecture (bronze → silver → gold) to incrementally process data, apply transformations, and create a star schema for analytics.

By leveraging Databricks **Autoloader, PySpark, Lakeflow Declarative Pipelines (Delta Live Tables), and dbt models**, the pipeline mimics real-world data engineering workflows while showcasing scalable, production-ready design principles. 

<img width="1445" height="898" alt="architecture" src="https://github.com/user-attachments/assets/8f7b663f-4f72-425a-8734-a94381717afa" />

## 🚀 Features

- **Incremental Ingestion (Bronze Layer)**  
 Data was incrementally ingested into the Bronze layer using Databricks Autoloader and Spark Structured Streaming. <br/>

<img width="1911" height="886" alt="Bronze_incremental_ingestion_job" src="https://github.com/user-attachments/assets/8bad346a-e0d1-4fc9-9013-4c360f66e634" /> <br/>


- **Transformations with Lakeflow (Silver Layer)**  
 Data cleaning and enrichment performed using Lakeflow Declarative Pipelines (formerly Delta Live Tables).
<img width="1916" height="881" alt="DLT_silver_pipeline" src="https://github.com/user-attachments/assets/bc7e0545-f42c-4dcc-8d8c-d075527918dc" />


- **Analytics-Ready Star Schema (Gold Layer)**  
Built Fact and Dimension tables in Databricks notebooks using dynamic Slowly Changing Dimensions (SCD).

- **Dynamic & Deployable Notebooks**  
  Modular notebooks that can be deployed easily in production environments.

- **Warehouse/Analytics Access**
  Ran business queries using dbt, storing results in a separate schema for reporting through Databricks SQL endpoints for business stakeholders.

  
### 🛠 Tech Stack

- **Platform**: Databricks (Free Edition)  
- **Data Processing:** PySpark, Autoloader, Lakeflow (Delta Live Tables)  
- **Modeling:** Dynamic SCD for Fact & Dimension tables  
- **Transformation/Analytics:** dbt (Data Build Tool) 
- **Architecture:** Medallion (Bronze → Silver → Gold)
- **Storage:** Delta Lake


### 💻 How to Run Locally

1. Clone this repo
2. Import notebooks into Databricks Free Edition.
3. Configure a Databricks cluster.
4. Upload raw source CSV files.
5. Execute pipeline in the following order:
Run Bronze ingestion (Autoloader) job
Run DLT Silver transformation (Lakeflow pipeline)
Run Gold dimension notebooks for each dimension
Run Gold fact notebook
Run dbt models (business queries) using dbt
6. Access the results in Databricks SQL Warehouse.


### 💡 Most Difficult Challenge

The biggest challenge was designing the **dynamic SCD logic** for gold tables and handling extensive data cleaning in Silver. Balancing scalability with maintainability inside Databricks notebooks took several iterations. If I had more time, I would further optimize dbt model performance and add **CI/CD for dbt + notebook deployment** to mimic real-world production workflows.
