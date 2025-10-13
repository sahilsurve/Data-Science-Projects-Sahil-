# ⚡ Real-Time Stocks Market Data Pipeline with Snowflake, Airflow & dbt
<br>

### 📌 Project Description

This project demonstrates a modern real-time data engineering pipeline built with **Apache Airflow, Kafka, MinIO (S3), Snowflake, and dbt**, orchestrated entirely through **Docker containers**.
The goal is to simulate real-time stock market data ingestion, transformation, and visualization using an automated, production-ready pipeline.

The data is streamed from the **Finnhub Stock API** into **Kafka**, stored in **MinIO** (S3-compatible object store), transformed through **dbt models in Snowflake** following a **Medallion Architecture** (Bronze → Silver → Gold), and finally visualized dynamically in **Power BI**.

<img width="1088" height="621" alt="Architecture" src="https://github.com/user-attachments/assets/d6232854-c990-4888-bd81-1d782544e633" />
<br>

## 🚀 Features

- **Real-time Data Streaming**  
Captures live stock quotes using Kafka producers and topics. <br/>

- **Data Lake Integration**  
Stores raw event data in MinIO (S3 bucket).

- **Automated Orchestration**  
Uses Airflow DAGs to automate extraction, loading, and transformation tasks

- **Modern Data Warehouse**  
 Snowflake serves as the centralized data warehouse for analytics.

- **Data Transformation with dbt** <br>
 Implements Medallion architecture — Bronze (raw), Silver (cleaned), Gold (aggregated) layers.

- **Dynamic BI Layer** <br>
 Power BI dashboard connected via DirectQuery for real-time updates.

- **Containerized Setup** <br>
 Entire stack (Airflow, Kafka, MinIO, Kafdrop, dbt) runs within Docker for reproducibility.
<br>
  
### 🛠 Tech Stack

- **Languages:** Python, SQL
- **Data Platform:** Snowflake, MinIO (S3) 
- **Real-time Streaming:** Apache Kafka, Kafdrop 
- **Workflow Orchestration:** Apache Airflow 
- **Transformations:** dbt (Data Build Tool) 
- **Architecture:** Medallion (Bronze → Silver → Gold)
- **Containerization:** Docker
- **Visualization:** Power BI
<br>

### 💻 How to Run Locally

1. Clone this repo and open it in VS Code.
2. Create and activate a virtual environment.
3. Install dependencies using pip install -r requirements.txt.
4. Copy docker-compose.yml to the infra folder and run docker compose up -d.
5. Initialize Airflow DB and create an admin user.
6. Access Airflow, Kafdrop, and MinIO UIs from the browser.
7. Get Finnhub API key and create a Kafka topic stock-quotes.
8. Run producer.py to stream real-time stock data into Kafka.
9. Create a bronze-transactions bucket in MinIO and run consumer.py to store data.
10. Configure Snowflake credentials in minio_to_snowflake.py and trigger the Airflow DAG.
11. Initialize dbt project and create Bronze, Silver, Gold models.
12. Run dbt run to build transformed views in Snowflake.
13. Connect Power BI to Snowflake (DirectQuery) and build dashboards.
<br>

### 📊 Final Deliverables

- Automated real-time data pipeline
- Snowflake tables (Bronze → Silver → Gold)
- Transformed analytics models with DBT
- Orchestrated DAGs in Airflow
- Power BI dashboard with live insights
<br>
<img width="1551" height="872" alt="image" src="https://github.com/user-attachments/assets/de252a66-bb70-4c63-8aa2-4a94ab0cd9e7" />
