# For Fact table bookings
# The goal is cleaning/transformation + validation/quality rules.
# No need for special handling of updates (data is mostly insert-only or just quality-checked).

from pyspark.sql.functions import *
import dlt  

# DLT streaming table reading raw bronze data
@dlt.table(
    name = 'stage_bookings'
) 
def stage_bookings():
    # Load raw streaming data from bronze volume path
    df = spark.readStream.format('delta')\
        .load('/Volumes/workspace/bronze/bronzevolume/bookings/data/')
    
    return df


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


# Set of rules
rules = {
    'rule1' : 'booking_id is not null',
    'rule2' : 'passenger_id is not null',
    'rule3' : 'flight_id is not null'
}


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


#############################################################################################################


# For dimension table flights
# The goal is handling UPSERTs / Slowly Changing Dimensions (SCD) so Silver always reflects the latest state of each flight.
# CDC means detecting and applying changes (inserts, updates, deletes) from source data into your target table.


# DLT view to read raw flights data from Bronze
@dlt.view(
    name = 'transformed_flights'
)
def transformed_flights():
    
    df = spark.readStream.format('delta')\
        .load('/Volumes/workspace/bronze/bronzevolume/flights/data/')

    return df

# Create an empty streaming Delta table for Silver layer
dlt.create_streaming_table('silver_flights')

# Define automatic CDC (Change Data Capture) flow
dlt.create_auto_cdc_flow(
  target = "silver_flights",        # Destination Silver table
  source = "transformed_flights",   # Upstream view as data source
  keys = ["flight_id"],             # Primary key for identifying rows
  sequence_by = col("flight_id"),
  stored_as_scd_type = 1            # SCD Type 1: overwrite old rows (no history)
)



#############################################################################################################

# For dimension table passengers (No transformations performed)

@dlt.view(
    name = 'transformed_passengers'
)
def transformed_passengers():
    
    df = spark.readStream.format('delta')\
        .load('/Volumes/workspace/bronze/bronzevolume/customers/data/')
    return df

dlt.create_streaming_table('silver_passengers')

dlt.create_auto_cdc_flow(
  target = "silver_passengers",       
  source = "transformed_passengers",   
  keys = ["passenger_id"],            
  sequence_by = col("passenger_id"),
  stored_as_scd_type = 1          
)


#############################################################################################################

# For dimension table aiports (No transformations performed)

@dlt.view(
    name = 'transformed_airports'
)
def transformed_airports():
    
    df = spark.readStream.format('delta')\
        .load('/Volumes/workspace/bronze/bronzevolume/airports/data/')
    return df

dlt.create_streaming_table('silver_airports')

dlt.create_auto_cdc_flow(
  target = "silver_airports",       
  source = "transformed_airports",   
  keys = ["airport_id"],            
  sequence_by = col("airport_id"),
  stored_as_scd_type = 1          
)
