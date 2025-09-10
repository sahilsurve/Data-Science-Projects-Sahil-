# Databricks notebook source
# MAGIC %sql
# MAGIC select * from workspace.silver.silver_flights

# COMMAND ----------

# MAGIC %md
# MAGIC Dimension tables have a primary key called as **dim surrogate key** which is artificially generated.
# MAGIC
# MAGIC **Slowly changing dimension**\
# MAGIC A dimension table has 5 records with surrogate keys 1,2,3,4,5. In new load data for record 2,3 is changed and also new data is added. So surrogate key for 2,3 should not change but only data is updated and new record must get key as 6. This is called Slowly changing dimension. By definition A slowly changing dimension (SCD) in data warehousing is a dimensional attribute that changes over time, requiring strategies to manage these changes and preserve historical data for analysis. Types are:
# MAGIC
# MAGIC - Type 1 (Update in Place): The old attribute value is simply overwritten with the new value (this project). 
# MAGIC - Type 2 (Preserve History): New records are created to capture the changed attribute, while the old record remains to preserve history. 
# MAGIC - Type 3 (Keep Previous Value): The current and previous values of an attribute are kept in the same dimension record. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Static approach using widgets**

# COMMAND ----------

# # key columns
# dbutils.widgets.text('keycols', '')

# # CDC columns
# dbutils.widgets.text('cdccols', '')

# # Back-dated refresh
# dbutils.widgets.text('backdated_refresh', '')

# # Source object
# dbutils.widgets.text('source_object', '')

# # Source schema
# dbutils.widgets.text('source_schema', '')

# # CDC column
# cdc_col = dbutils.widgets.get('cdccols')

# # Key columns list
# key_cols = dbutils.widgets.get('keycols')
# key_cols_list = eval(key_cols)                # need to pass a value in keycols widget else will throw error

# # Back-dated refresh
# backdated_refresh = dbutils.widgets.get('backdated_refresh')

# # Source object
# source_object = dbutils.widgets.get('source_object')

# # Source schema
# source_schema = dbutils.widgets.get('source_schema')

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Parameters**

# COMMAND ----------

# CDC column                        #  column used to track and manage changes to data in a database
cdc_col = 'modifiedDate'

# Key columns list
key_cols = "['passenger_id']"          # Primary key for flights
key_cols_list = eval(key_cols)               

# Back-dated refresh
backdated_refresh = ""

# Source object                     # flights table name in silver
source_object = 'silver_passengers'

# Source schema
source_schema = 'silver'

# Target object                     
target_object = 'DimPassengers'

# Target schema
target_schema = 'gold'

# Surrogate key 
surrogate_key = 'DimPassengersKey'


# COMMAND ----------

# MAGIC %md
# MAGIC ### **Incremental Data Ingestion**

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Last load date**
# MAGIC

# COMMAND ----------

# The purpose of this code is to determine the last_load timestamp for an incremental load process, based on whether a table already exists or whether a backdated refresh is requested.

# Check if backdated refresh is not provided (empty string)
if backdated_refresh == "":

  # If the target table already exists in the workspace
  if spark.catalog.tableExists(f'workspace.{target_schema}.{target_object}'):

    # Get the maximum modifiedDate (latest change) from the target table
    last_load = spark.sql(f'select max({cdc_col}) from workspace.{target_schema}.{target_object}').collect()[0][0]
  
  # If the target table does not exist
  else : 
    # Set last_load to a very old default date (acts as a starting point)
    last_load = '1900-01-01 00:00:00'

# If backdated refresh is provided
else:
  # Use the given backdated refresh timestamp as last_load
  last_load = backdated_refresh

last_load  

# COMMAND ----------

df_src = spark.sql(f'select * from workspace.{source_schema}.{source_object} where {cdc_col} >= "{last_load}"')

df_src.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Old vs New records
# MAGIC
# MAGIC This Spark code is handling a scenario where it checks whether a target table exists and then builds a DataFrame (df_target) either by selecting from that table (if it exists) or by creating an empty DataFrame with the same schema (if it doesn’t exist).
# MAGIC

# COMMAND ----------

# Check if the target table exists in the given schema
if spark.catalog.tableExists(f'workspace.{target_schema}.{target_object}'):

    # Join key columns into a comma-separated string for SQL query (incremental load case)
    key_cols_string_incremental = ', '.join(key_cols_list)

    # Select key columns, surrogate key, create_date, and update_date from the existing target table
    df_target = spark.sql(f' select {key_cols_string_incremental}, {surrogate_key}, create_date, update_date from workspace.{target_schema}.{target_object}')

# if target table doesnt exist
else:

    # Build placeholder expressions ('' as column_name) for key columns (initial load case)
    key_cols_string_init = [f"'' as {i}" for i in key_cols_list]
    # Join placeholder expressions into a comma-separated string
    key_cols_string_init = ', '.join(key_cols_string_init)

    # Create an empty DataFrame with the same schema 
    # where condition is applied to get no records
    df_target = spark.sql(f"select {key_cols_string_init}, cast('0' as int) as {surrogate_key}, cast('1900-01-01 00:00:00' as timestamp) as create_date, cast('1900-01-01 00:00:00' as timestamp) as update_date where 1=0")

df_target.display()

# COMMAND ----------

# MAGIC %md
# MAGIC **Join condition**

# COMMAND ----------

join_condition = ' and '.join([f"src.{i} = trg.{i}" for i in key_cols_list])

# COMMAND ----------

df_src.createOrReplaceTempView("src")
df_target.createOrReplaceTempView("trg")

df_join = spark.sql(f"""
            select  src.*,
                    trg.{surrogate_key}, trg.create_date, trg.update_date
            from src
            left join trg
            on {join_condition} """)

# COMMAND ----------

from pyspark.sql.functions import *

# Old records
df_old = df_join.filter(col(f'{surrogate_key}').isNotNull())

# New records
df_new = df_join.filter(col(f'{surrogate_key}').isNull())

df_old.display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### **Enriching dataframes**

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Preparing df_old_enriched**

# COMMAND ----------

df_old_enriched = df_old.withColumn('update_date', current_timestamp())

df_old_enriched.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Preparing df_new_enriched**
# MAGIC
# MAGIC This block is handling surrogate key generation and enrichment of a new DataFrame depending on whether the target table exists or not.

# COMMAND ----------

# Check if the target table exists in the given schema
if spark.catalog.tableExists(f"workspace.{target_schema}.{target_object}"):

    # Get the current maximum surrogate key from the existing target table
    max_surrogate_key = spark.sql(f"""
                                  select max({surrogate_key}) from workspace.{target_schema}.{target_object}
                                  """).collect()[0][0]
    
    # If target table exists, start surrogate keys from the next available number (incremental load case)
    df_new_enriched = df_new.withColumn(f'{surrogate_key}', lit(max_surrogate_key + 1 + monotonically_increasing_id())).withColumn('create_date', current_timestamp()).withColumn('update_date', current_timestamp())

else:
    # If target table doesn’t exist, start surrogate keys from 0 (initial load case)
    max_surrogate_key = 0
    df_new_enriched = df_new.withColumn(f'{surrogate_key}', lit(max_surrogate_key + 1 + monotonically_increasing_id())).withColumn('create_date', current_timestamp()).withColumn('update_date', current_timestamp())
    
df_new_enriched.display()    

# COMMAND ----------

max_surrogate_key

# COMMAND ----------

df_old_enriched.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Unioning old and new records**

# COMMAND ----------

# unionbyname appends the table wrt the schema of the first table
df_union = df_old_enriched.unionByName(df_new_enriched)
df_union.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### UPSERT (update + insert)
# MAGIC This code checks if a Delta table exists in Spark, and then either merges new data into it (upsert logic) or creates it if it doesn’t exist.

# COMMAND ----------

from delta.tables import DeltaTable

# COMMAND ----------

if spark.catalog.tableExists(f"workspace.{target_schema}.{target_object}"):
    
    # Get a reference to the existing Delta table
    dlt_obj = DeltaTable.forName(spark, f"workspace.{target_schema}.{target_object}")

    # Start a merge (upsert) between source dataframe and target Delta table using surrogate key
    ( dlt_obj.alias("trg").merge(df_union.alias("src"), f"trg.{surrogate_key} = src.{surrogate_key}")\
            # Update if a match is found and only if source record is newer (based on CDC column)
            .whenMatchedUpdateAll(condition = f"src.{cdc_col} >= trg.{cdc_col}")\
            # Insert the row if it doesn’t exist in the target    
            .whenNotMatchedInsertAll()\
            .execute() )

 # If the target Delta table doesn’t exist
else:

    ( df_union.write.format("delta")\
        # Append mode (creates the table if it doesn’t exist)
        .mode("append")\
        # Save the dataframe as a new managed Delta table as gold layer in the target schema
        .saveAsTable(f"workspace.{target_schema}.{target_object}") )               

# COMMAND ----------

spark.sql(f"select * from workspace.{target_schema}.{target_object}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC Change the parameters for each dimension and run the notebook again. The data will be stored as delta table in the gold schema

# COMMAND ----------

# MAGIC %md
# MAGIC You can upload the SCD file in the raw volume and run the bronze job to see the incremental load and then run the silver pipeline to see the SCD logic and then run the gold notebook to see the final table with surrogate key and updated data with date