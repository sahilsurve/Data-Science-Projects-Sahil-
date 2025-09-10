# Databricks notebook source
# Catalog name                       
catalog = 'workspace'

# CDC column                        #  column used to track and manage changes to data in a database
cdc_column = 'modifiedDate'
               
# Back-dated refresh
backdated_refresh = ""

# Source object                     
source_object = 'silver_bookings'

# Source schema
source_schema = 'silver'

# Target object                     
target_object = 'FactBookings'

# Target schema
target_schema = 'gold'

# Source fact table 
fact_table = f'{catalog}.{source_schema}.{source_object}'

# Fact key columns list (booking date was added as 2 records had same info just different booking date)
fact_key_cols = ['DimPassengersKey', 'DimFlightsKey', 'DimAirportsKey', 'booking_date']


# COMMAND ----------

dimensions = [
    {
        "table" : f'{catalog}.{target_schema}.dimpassengers',
        "alias" : 'dimpassengers',
        "join_keys" : [('passenger_id', 'passenger_id')]     # (fact_col, dim_col)
    },

    {
        "table" : f'{catalog}.{target_schema}.dimflights',
        "alias" : 'dimflights',
        "join_keys" : [('flight_id', 'flight_id')]     # (fact_col, dim_col)
    },

    {
        "table" : f'{catalog}.{target_schema}.dimairports',
        "alias" : 'dimairports',
        "join_keys" : [('airport_id', 'airport_id')]     # (fact_col, dim_col)
    }
]

# Columns you want to keep from fact table (numeric & date columns)
fact_columns = ['amount', 'booking_date', 'modifiedDate']

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Last load date**

# COMMAND ----------

if backdated_refresh == "":

  # If target table exists in destination
  if spark.catalog.tableExists(f'{catalog}.{target_schema}.{target_object}'):

    # Get the maximum modifiedDate (latest change) from the target table
    last_load = spark.sql(f'select max({cdc_column}) from {catalog}.{target_schema}.{target_object}').collect()[0][0]
  
  # If the target table does not exist
  else : 
    
    last_load = '1900-01-01 00:00:00'

# If backdated refresh is provided
else:
  # Use the given backdated refresh timestamp as last_load
  last_load = backdated_refresh

last_load  

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Dynamic Fact query [Bring keys]**

# COMMAND ----------

def generate_fact_query_incremental (fact_table, dimensions, fact_columns, cdc_columns, processing_date):
    fact_alias = "f"

    # Base columns to select
    select_cols = [f'{fact_alias}.{col}' for col in fact_columns]

    # build joins dynamically
    join_clauses = []
    for dim in dimensions:
        table_full = dim['table']
        alias = dim['alias']
        table_name = table_full.split('.')[-1]
        surrogate_key = f'{alias}.{table_name}Key'
        select_cols.append(surrogate_key)

        # Build On clause
        on_condition = [f'{fact_alias}.{fk} = {alias}.{dk}' for fk, dk in dim['join_keys']]

        join_clause = f"LEFT JOIN {table_full} {alias} ON " + " AND ".join(on_condition)
        join_clauses.append(join_clause) 


    # Final select and join clause
    select_clause = ",\n       ".join(select_cols)
    joins = "\n".join(join_clauses)

    # Where clause for incremental filtering
    where_clause = f"{fact_alias}.{cdc_columns} >= DATE('{last_load}')"

    # Final query
    query = f"""
    SELECT
        {select_clause}
    FROM {fact_table} {fact_alias} {joins}
    WHERE 
        {where_clause}
        """.strip()

    return query   

# COMMAND ----------

query = generate_fact_query_incremental(fact_table, dimensions, fact_columns, cdc_column, last_load)
print(query)

# COMMAND ----------

# MAGIC %md
# MAGIC #### **DF_Fact**

# COMMAND ----------

df_fact = spark.sql(query)

# COMMAND ----------

df_fact.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Upsert**

# COMMAND ----------

# Fact key columns merge condition
fact_key_cols_str = " AND ".join([f"src.{col} = trg.{col}" for col in fact_key_cols])
fact_key_cols_str

# COMMAND ----------

from delta.tables import DeltaTable

if spark.catalog.tableExists(f"{catalog}.{target_schema}.{target_object}"):
    
    # Get a reference to the existing Delta table
    dlt_obj = DeltaTable.forName(spark, f"{catalog}.{target_schema}.{target_object}")

    # Start a merge (upsert) between source dataframe and target Delta table using surrogate key
    ( dlt_obj.alias("trg").merge(df_fact.alias("src"), fact_key_cols_str)\
            # Update if a match is found and only if source record is newer (based on CDC column)
            .whenMatchedUpdateAll(condition = f"src.{cdc_column} >= trg.{cdc_column}")\
            # Insert the row if it doesn’t exist in the target    
            .whenNotMatchedInsertAll()\
            .execute() )

 # If the target Delta table doesn’t exist
else:

    ( df_fact.write.format("delta")\
        # Append mode (creates the table if it doesn’t exist)
        .mode("append")\
        # Save the dataframe as a new managed Delta table as gold layer in the target schema
        .saveAsTable(f"{catalog}.{target_schema}.{target_object}") )               

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from workspace.gold.factbookings

# COMMAND ----------

# MAGIC %md
# MAGIC No duplicate records should be present in any dimension table. To verify we can run below code for each dimension

# COMMAND ----------

from pyspark.sql.functions import *
df= spark.sql(f"select * from {catalog}.{target_schema}.dimpassengers").groupBy('dimpassengersKey').count().filter(col('count') > 1)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **DBT**
# MAGIC
# MAGIC dbt (data build tool) is primarily used for the transformation step in ETL/ELT pipelines. It follows a SQL-first approach, which makes writing and debugging transformations easier compared to code-heavy frameworks like PySpark. In this project, we will use dbt to create curated business views that will be consumed by stakeholders which were saved in dbt_sahilsurve schema.

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Countries and their respective total amount sorted in descending order
# MAGIC select * from workspace.dbt_sahilsurve.countries_and_amount