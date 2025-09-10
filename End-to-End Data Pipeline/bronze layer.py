# Databricks notebook source
# MAGIC %md
# MAGIC ##  **Incremental Data Ingestion**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC In bronze layer raw data is stored in Volumes because you just need a landing zone / data lake folder to dump data in. No real transformations, just “land and store.”

# COMMAND ----------

# Tell Spark to read streaming data using Auto Loader (cloudFiles)
df = (spark.readStream.format('cloudfiles')
    # The incoming files are in CSV format
    .option('cloudfiles.format', 'csv')
    # Store and track the data schema in this checkpoint folder so Spark knows column structure
    .option('cloudfiles.schemaLocation', '/Volumes/workspace/bronze/bronzevolume/bookings/checkpoint')
    # If new or unexpected columns appear, put them into a special _rescued_data column instead of failing
    .option('cloudfiles.schemaEvolutionMode', 'rescue')
    # Start reading new files as they arrive from this raw cloud storage folder
    .load('/Volumes/workspace/raw/rawvolume/rawdata/bookings/'))
   

# COMMAND ----------

# Write the streaming data out in Delta Lake format
(df.writeStream.format('delta')
    # Add new rows to the table without modifying existing data (append-only mode)
    .outputMode('append')
    # Process all available new files once and then stop
    .trigger(once=True)
    # Save the processed Delta data into this bronze storage path
    .option('path', '/Volumes/workspace/bronze/bronzevolume/bookings/data')
    # Keep track of which files have already been processed in this checkpoint folder (to avoid duplicates and resume on failure)
    .option('checkpointLocation', '/Volumes/workspace/bronze/bronzevolume/bookings/checkpoint')
    # Start the Auto Loader streaming job
    .start())

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Checking the loaded data
# MAGIC select * from delta.`/Volumes/workspace/bronze/bronzevolume/bookings/data/`

# COMMAND ----------

# MAGIC %md
# MAGIC Uploaded the incremented booking data into raw volume

# COMMAND ----------

# Writing the new incremented booking data into Delta Lake (bronze layer)
df.writeStream.format('delta')\
.outputMode('append')\
.trigger(once=True)\
.option('path', '/Volumes/workspace/bronze/bronzevolume/bookings/data')\
.option('checkpointLocation', '/Volumes/workspace/bronze/bronzevolume/bookings/checkpoint')\
.start()



# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check if the bronze booking table is incremented
# MAGIC select * from delta.`/Volumes/workspace/bronze/bronzevolume/bookings/data/`

# COMMAND ----------

# MAGIC %md
# MAGIC We observed additional newly ingested rows (300 rows appended in this run) \
# MAGIC Shows that autoloader is working fine

# COMMAND ----------

# MAGIC %md
# MAGIC However this static process needs to be done for all other files too. So instead we will use a dynamic approach. We will generate a widget box whose input will ingest the folder data into bronze layer

# COMMAND ----------

# Create a widget box with name src
dbutils.widgets.text("src", "")

# Display the widget box's inputted value
src_value = dbutils.widgets.get("src")
src_value

# COMMAND ----------

df = (spark.readStream.format('cloudfiles')
    .option('cloudfiles.format', 'csv')
    # Using dynamic value from the widget box
    .option('cloudfiles.schemaLocation', f'/Volumes/workspace/bronze/bronzevolume/{src_value}/checkpoint')
    .option('cloudfiles.schemaEvolutionMode', 'rescue')
    .load(f'/Volumes/workspace/raw/rawvolume/rawdata/{src_value}/'))

# Writing the inputted value's (airports) data into bronze layer
df.writeStream.format('delta')\
.outputMode('append')\
.trigger(once=True)\
.option('path', f'/Volumes/workspace/bronze/bronzevolume/{src_value}/data')\
.option('checkpointLocation', f'/Volumes/workspace/bronze/bronzevolume/{src_value}/checkpoint')\
.start()    

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check if airports data is ingested into the bronze booking table 
# MAGIC select * from delta.`/Volumes/workspace/bronze/bronzevolume/airports/data/`

# COMMAND ----------

# MAGIC %md
# MAGIC We observed that the airports data was ingested successfully. However this process is still static as we need to manually type the value. Instead we will create another notebook which has source parameters to pass in here. 