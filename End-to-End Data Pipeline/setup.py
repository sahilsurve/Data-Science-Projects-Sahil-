# Databricks notebook source
# MAGIC %md
# MAGIC - Created schema raw
# MAGIC - Created volume in that called rawvolume
# MAGIC - Created directory in that called rawdata
# MAGIC - Created subfolders for each dimension and a fact table
# MAGIC - Uploaded csv files in respective folders
# MAGIC - Created schemas for bronze, silver and gold as \
# MAGIC Each layer provides specific data processing, validation, and structure, which incrementally improves data quality and reliability as it moves through each layer in the pipeline. The separation allows for changes and new transformations in the Silver and Gold layers without affecting the raw data in the Bronze layer, making the system more adaptable to evolving business needs. 
# MAGIC - Created volumes for each bronze, silver and gold layers

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from delta.`/Volumes/workspace/bronze/bronzevolume/customers/data/`