# This Python script continuously retrieves live stock price quotes for 5 major companies from the Finnhub API and sends (produces) 
# those quotes to a Kafka topic (stock-quotes).
# It does this in an infinite loop, fetching new data every 6 seconds.


#Import requirements
import time
import json
import requests
from kafka import KafkaProducer


# Define API connection variables

API_KEY = '<your api key>'            # API key retrieved from website and its url + v1/quote
BASE_URL = "https://finnhub.io/api/v1/quote"
SYMBOLS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]             # List of stock symbols to track 


#Initialize Kafka Producer
producer = KafkaProducer (
    bootstrap_servers=["host.docker.internal:29092"],           # Use this when running VS Code outside Docker; use 'localhost' if running inside Docker
    value_serializer=lambda v: json.dumps(v).encode("utf-8")    # Serialize python dictionary into json and then encode it into bytes
)


#Function to fetch a stock quote from the API
def fetch_quote(symbol):
    url = f"{BASE_URL}?symbol={symbol}&token={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        data["symbol"] = symbol                             # Adding extra column (metadata) for symbols 
        data["fetched_at"] = int (time.time())              # Adding extra column for current time (UTC)
        return data
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None
    

#Main loop: fetch and produce stock quotes continuously
while True:
    for symbol in SYMBOLS:
        quote = fetch_quote(symbol)                     # Fetching current quote for every symbol from API
        if quote:
            print(f"Producing: {quote}")
            producer.send("stock-quotes", value=quote)  # Send message to kafdrop topic
    time.sleep(6)                                       # Retrieve records for all symbols once very 6 secs
