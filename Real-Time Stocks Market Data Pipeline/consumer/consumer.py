#This script consumes real-time stock quote messages from a Kafka topic (stock-quotes) and 
#saves each message as a JSON file in a MinIO S3 bucket (bronze-transactions).
# It ensures the MinIO bucket exists, continuously listens to the Kafka topic, and writes each 
# message to object storage using a folder structure based on the stock symbol and timestamp.


#Import required libraries
import json
import boto3
import time
from kafka import KafkaConsumer

#Initialize MinIO (S3-compatible) client
s3 = boto3.client(
    "s3",                                   # MinIO endpoint, access and secret key can be checked from yml file 
    endpoint_url="http://localhost:9002",
    aws_access_key_id="admin",
    aws_secret_access_key="password123"
)

bucket_name = "bronze-transactions"


# Ensure the target bucket exists (idempotent check)
try:
    s3.head_bucket(Bucket=bucket_name)
    print(f"Bucket {bucket_name} already exists.")
except Exception:
    s3.create_bucket(Bucket=bucket_name)
    print(f"Created bucket {bucket_name}.")


#Define Kafka Consumer configuration
consumer = KafkaConsumer(
    "stock-quotes",                                             #Topic to consume from
    bootstrap_servers=["host.docker.internal:29092"],
    auto_offset_reset="earliest",                               # Start from earliest message if no offset is committed (consider old files as well)
    enable_auto_commit=True,                                    # Automatically commit offsets
    group_id="bronze-consumer1",                                # Consumer group name
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))  # Deserialize JSON from bytes
)

print("Consumer streaming and saving to MinIO...")


# Main loop: continuously consume messages and save to MinIO
for message in consumer:
    record = message.value                                  # Extract message content (stock quote)
    symbol = record.get("symbol", "unknown")                # Get stock symbol
    ts = record.get("fetched_at",int(time.time()))
    key = f"{symbol}/{ts}.json"                             # S3 object key (folder = symbol, file = timestamp)

    # Upload record to MinIO bucket as a JSON object
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(record),
        ContentType="application/json"
    )
    print(f"Saved record for {symbol} = s3://{bucket_name}/{key}")