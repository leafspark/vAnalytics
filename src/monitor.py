import requests
import time
import os
import json
from datetime import datetime
import logging
import zstd
import sqlite3

print("Starting monitor.")

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="monitor.log",  # Log to a file named monitor.log
    filemode="a",
)  # Append to the log file
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Model information
models = {
    "Example-Model-22B-FP8-dynamic": "http://112.83.15.44:8883",
    "Mistral-7B-bf16": "http://57.214.142.199:8090",
}


def call_metrics_endpoint(model_name, base_url):
    url = f"{base_url}/metrics"
    logging.debug(f"Calling metrics endpoint: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logging.debug(f"Received successful response from {url}")
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling {url}: {e}")
        return f"Error calling {url}: {e}"


def normalize_metrics(metrics_data):
    """Normalizes the metrics data from vLLM."""
    normalized_data = {}
    lines = metrics_data.strip().split("\n")
    for line in lines:
        if line.startswith("#"):  # Ignore comment lines
            continue
        parts = line.split(" ")
        metric_name = parts[0]
        metric_value_str = parts[1]

        # Try to convert to decimal, otherwise keep as string
        try:
            metric_value = float(metric_value_str)
            if metric_name.endswith("_total") or metric_name.endswith("_count"):
                metric_value = int(metric_value)
            elif "e+" in metric_value_str or "e-" in metric_value_str:
                metric_value = "{:.10f}".format(metric_value)
        except ValueError:
            metric_value = metric_value_str

        # Extract labels from metric name
        if "{" in metric_name:
            metric_name, labels_str = metric_name[:-1].split("{")
            labels = {}
            for label_pair in labels_str.split(","):
                key, value = label_pair.split("=")
                labels[key.strip('"')] = value.strip('"')
            if metric_name not in normalized_data:
                normalized_data[metric_name] = []
            normalized_data[metric_name].append(
                {"labels": labels, "value": metric_value}
            )
        else:
            normalized_data[metric_name] = metric_value

    return normalized_data


def log_response(model_name, response_data):
    timestamp = int(datetime.now().timestamp())
    normalized_data = normalize_metrics(response_data)

    db_filename = f"./data/{model_name}.sqlite"
    os.makedirs(os.path.dirname(db_filename), exist_ok=True)

    max_retries = 3
    for retry in range(max_retries):
        try:
            conn = sqlite3.connect(db_filename)
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS json_data
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             data TEXT NOT NULL,
             timestamp INTEGER NOT NULL)
            """
            )

            # Insert the data
            cursor.execute(
                "INSERT INTO json_data (data, timestamp) VALUES (?, ?)",
                (json.dumps(normalized_data), timestamp),
            )

            conn.commit()
            conn.close()

            logging.debug(f"Saved metrics data to {db_filename}")
            break  # Exit the retry loop if successful
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logging.warning(
                    f"Database locked for {model_name}, retrying in 5 seconds... (Attempt {retry+1}/{max_retries})"
                )
                time.sleep(5)  # Wait before retrying
            else:
                logging.error(f"Error writing to database for {model_name}: {e}")
                break  # Exit the retry loop for other errors


while True:
    for model_name, base_url in models.items():
        response_data = call_metrics_endpoint(model_name, base_url)
        if response_data and not response_data.startswith(
            "Error"
        ):  # Check for valid data
            logging.info(f"Metrics for {model_name} valid")  # Log metrics to console
        log_response(model_name, response_data)

    logging.debug("Waiting for 30 seconds...")
    time.sleep(30)
