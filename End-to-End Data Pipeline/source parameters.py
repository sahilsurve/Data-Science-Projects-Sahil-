# Databricks notebook source
# An array of dictionaries of source data folders
src_array = [

    {'src' : 'bookings'},
    {'src' : 'airports'},
    {'src' : 'customers'},
    {'src' : 'flights'}

]

# COMMAND ----------

# Stores src_array under the key 'output_key' so other tasks in the same Databricks job can retrieve it
dbutils.jobs.taskValues.set(key = 'output_key', value= src_array)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will create a task called Parameters which has path of this notebook and takes the parameter values from here. 
# MAGIC
# MAGIC Then we will join another task called Incremental_ingestion to Parameters task with path as the bronze layer notebook. This will fetch the parameter values. But since we have multiple values as dictionaries in an array we will put the task in for loop with key as 'src' as seen in dictionary and value as '{{input.src}}' as it represents iteration for each value in the array. Eg In python : for i.key in array print(value) of each dictionary that will be our folders names  

# COMMAND ----------

# MAGIC %md
# MAGIC After creating the workflow, run it. The airports and bookings table wont be populated as it was already ingested. However the customers and flights tables will be populated.

# COMMAND ----------

# MAGIC %md
# MAGIC Then upload the incremental data to the source folder and run the workflow again.

# COMMAND ----------

# MAGIC %md
# MAGIC The end data is stored in data lake tables. Same would be for silver and gold