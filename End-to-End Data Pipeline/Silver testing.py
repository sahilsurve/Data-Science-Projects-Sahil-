# Databricks notebook source
# MAGIC %md
# MAGIC **This notebook is just for info about the DLT pipeline for silver layer. Not to use anywhere**

# COMMAND ----------

# MAGIC %md
# MAGIC Silver (and Gold) layer require transformations, schema enforcement, and data quality checks.\
# MAGIC DLT (Delta Live Tables) is designed exactly for this:
# MAGIC
# MAGIC - Lets you declare transformations as code (@dlt.table, @dlt.view)
# MAGIC - Manages streaming ingestion automatically
# MAGIC - Enforces quality rules 
# MAGIC - Tracks lineage from bronze → silver → gold
# MAGIC - Handles schema evolution, checkpoints, retries, and job orchestration automatically
# MAGIC
# MAGIC That’s why unlike bronze where vloumes are used, your silver is declared in DLT because it’s not just storage, it’s a data pipeline with transformations + quality enforcement.

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

# This chunk of code is for testing purposes only. To see if the transformations are working as expected, we can use the following code.

"""
# Read the bronze volume data for bookings into a Spark DataFrame
df = spark.read.format('delta')\
    .load('/Volumes/workspace/bronze/bronzevolume/bookings/data/')

display(df)    

# Add the current timestamp to the booking data, drop null column '_rescued_data' and cast 'amount' to double and format of 'booking_date'
df = df.withColumn('modifiedDate', current_timestamp()).drop('_rescued_data')\
    .withColumn('amount', col('amount').cast('double'))\
    .withColumn('booking_date', to_date(col('booking_date')))
           

display(df) """

# COMMAND ----------

# MAGIC %md
# MAGIC The DLT pipeline which we will create now cannot run on this cluster. Hence we will first generate the logic here and then debug it in different cluster

# COMMAND ----------

# Will throw error
import dlt  

# COMMAND ----------

# We can create three things in DLT ie Streaming tables, Mat views and view - streaming

# DLT streaming table reading raw bronze data
@dlt.table(
    name = 'stage_bookings'
) 
def stage_bookings():
    # Load raw streaming data from bronze volume path
    df = spark.readStream.format('delta')\
        .load('/Volumes/workspace/bronze/bronzevolume/bookings/data/')()
    
    return df

# COMMAND ----------

# DLT view to apply transformations before Silver
@dlt.view(
    name = 'transformed_bookings'
)
def transformed_bookings():
# Can also use df = dlt.read('stage_bookings')    
    df = spark.readStream.table('stage_bookings')
    # Add required transformations
    df = df.withColumn('modifiedDate', current_timestamp()).drop('_rescued_data')\
    .withColumn('amount', col('amount').cast('double'))\
    .withColumn('booking_date', to_date(col('booking_date')))

    return df


# COMMAND ----------

# MAGIC %md
# MAGIC Why use a @dlt.view between tables?
# MAGIC
# MAGIC 1. Separation of logic
# MAGIC
# MAGIC stage_bookings (table) → raw ingest from Bronze.
# MAGIC transformed_bookings (view) → apply lightweight transformations (casts, renames, formatting).
# MAGIC silver_bookings (table) → apply business rules + data quality checks.
# MAGIC
# MAGIC This way, raw ingestion, transformation, and validation are decoupled steps.
# MAGIC
# MAGIC 2. Reusability : The view transformed_bookings can be reused by other downstream tables.
# MAGIC
# MAGIC Example: maybe both silver_bookings and revenue need the same cleaned amount column — instead of repeating the code, they can both read from the view.
# MAGIC
# MAGIC 3. Lineage and debugging
# MAGIC
# MAGIC DLT automatically tracks lineage. Having an intermediate view makes it easier to see where data was transformed vs. where rules were applied. If something goes wrong, you can debug at the view level before rules drop data.

# COMMAND ----------

# Set of rules
rules = {
    'rule1' : 'booking_id is not null',
    'rule2' : 'passenger_id is not null',
    'rule3' : 'flight_id is not null'
}

# COMMAND ----------

# Final curated Silver table
@dlt.table(
    name = 'silver_bookings'
)

# Pass all the rules on the final table. If any record contradicts then 3 things can happen:
# It will throw warning (default), fail the job (expect_all_or_fail) or drop the record (expect_all_or_drop)
# We will drop invalid records here
@dlt.expect_all_or_drop(rules)

def silver_bookings():
    # Read from transformed view and materialize as Silver table
    df = spark.readStream.table('transformed_bookings')
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC Then go to Pipelines tab and create new ETL pipeline called 'DLT_silver_pipeline'

# COMMAND ----------

# MAGIC %md
# MAGIC Previously for DLT we used to give path of this notebook for creation of DLT pipeline.
# MAGIC However since the latest upgrade to lakeflow declarative pipeline we got two new options:
# MAGIC 1. start with single transformation/ empty file: in which a blank transformation.py file will be opened and we need to copy all the above code into it. (Will use this)
# MAGIC
# MAGIC 2. add existing assests: in which we will need python file of our notebook. So we need to convert our notebook to python file 

# COMMAND ----------

# MAGIC %md
# MAGIC So execute dry run on the my_transformation.py and you will see a DAG of our lakeflow pipeline with 3 stages.
# MAGIC If successful, run pipeline and silver layer for bookings will complete. You can obeserve the number of output records. If you run the pipeline again then output records will be zero as the table is streamed and only new data is read.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC A folder with the pipeline's name containing transformation.py will be generated. The goal was cleaning + validation and there was No need for special handling of updates (data is mostly insert-only or just quality-checked). 
# MAGIC Similarly add code for the dimension tables (flights, customers and airports ). Here, the goal is handling UPSERTs / Slowly Changing Dimensions (SCD). Change data capture (CDC) makes sure your Silver dimension tables are always synchronized with changes from the raw bronze data, instead of just appending new rows like bookings.
# MAGIC Run the pipeline and observe all tables joiined in the silver business table. 

# COMMAND ----------

# MAGIC %md
# MAGIC If you run the pipeline again, no records will be processed unless new data has arrived. This property of producing the same result on repeated runs is called idempotency.

# COMMAND ----------

# MAGIC %md
# MAGIC The end data is stored in Streaming tables. They are a specific type of Unity Catalog managed tables designed for incremental data processing and low-latency streaming. They include processing logic defined by "flows" within a pipeline, typically in Delta Live Tables (DLT) or Lakeflow Declarative Pipelines. Streaming tables and managed tables are both delta tables.