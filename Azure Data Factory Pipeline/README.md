# ⚡ Azure Data Factory End-to-End Data Engineering Project
<br>

### 📌 Project Description

This project demonstrates an end-to-end data engineering pipeline built on **Azure Data Factory (ADF)**, integrating data from **on-premises systems, web APIs, and Azure SQL databases** into a scalable Data Lakehouse architecture (Bronze → Silver → Gold layers).

It automates ingestion, transformation, and orchestration processes using ADF and **Logic Apps** for alerts, showcasing core skills in ETL/ELT, cloud migration, data modelling, and **Azure DevOps** readiness.

**Goal:** Automate ingestion and transformation from multiple sources to deliver analytics on top-performing airlines and total revenue insights.

<img width="1865" height="1000" alt="Architecture" src="https://github.com/user-attachments/assets/3e9b6fc6-5518-44f9-9aa6-9a5e13ae55ac" />
<br>

## 🚀 Key Components

- On-Premises to Azure Migration using Self-Hosted Integration Runtime  
- Web API Data Ingestion from live JSON endpoints
- Incremental SQL Data Loading using watermarking logic
- ADF Orchestration Pipelines to automate and sequence data flows
- Pyspark Transformations (Data Flows) to build Silver and Gold layers
- Delta Lake Upserts for efficient merge and update operations
- Logic App Alerts for automated pipeline failure email notifications
- Azure DevOps Integration for version control
<br>
  
### 🏗️ Architecture Workflow

**1. Bronze Layer – Raw Data Ingestion** 
- On-prem CSV → Azure Data Lake (ADLS Gen2) via Self-Hosted Integration Runtime
- Web API → ADLS in JSON format
- Azure SQL Database → ADLS with incremental loads <br>

**2. Silver Layer – Data Transformation** 
- PySpark dataflows for cleansing and normalization
- Delta Lake Upserts for data consistency
- Output stored in “silver” container

<img width="1918" height="1091" alt="Silver Data flow" src="https://github.com/user-attachments/assets/2dcf2dfc-a4c5-4f93-90a7-53f2700a204d" /><br>


**3. Gold Layer – Business Views**
- Aggregations, joins, and analytics-ready datasets
- Example: Top 10 airlines and airports by total ticket sales
<img width="1918" height="1091" alt="Gold layer business views" src="https://github.com/user-attachments/assets/209d7e02-94c7-47f5-b6b8-4b7d424827e5" />

<br>

### 🏁 Outcome

- Automated ingestion from 3 different data sources
- Dynamic, scalable data pipelines with parameterization
- End-to-end orchestration and monitoring in ADF
- Structured Lakehouse model ready for Power BI or Synapse analytics

<img width="1481" height="442" alt="image" src="https://github.com/user-attachments/assets/619ba21a-5670-414c-82f2-82aa200e8f6e" />



### 📚 Future Enhancements

- Integrate **Databricks notebooks** for complex transformations
- Add **Power BI dashboards** for gold layer insights
- Deploy pipelines using **ADF ARM templates** via Azure DevOps
