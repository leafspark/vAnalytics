import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.offline as pyo
from flask import Flask, render_template, request, send_file
from plotly.subplots import make_subplots

from globals import load_env


load_env()

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 4421))
DATA_DIR = os.getenv("DATA_DIR", "./data")
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 30))
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Global variable to store the cached data
cached_data = {}
last_modified_times = {}


async def load_data_from_db(filepath):
    global cached_data

    try:
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()

        twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
        timestamp_24h_ago = int(twenty_four_hours_ago.timestamp())

        cursor.execute(
            "SELECT data, timestamp FROM json_data WHERE timestamp >= ?",
            (timestamp_24h_ago,),
        )
        rows = cursor.fetchall()

        model_name = os.path.splitext(os.path.basename(filepath))[0]

        # Optimize data structure creation
        new_data = {}
        for row in rows:
            data = json.loads(row[0])
            timestamp = datetime.fromtimestamp(row[1])

            for metric_name, metric_data in data.items():
                if metric_name not in new_data:
                    new_data[metric_name] = {}
                if model_name not in new_data[metric_name]:
                    new_data[metric_name][model_name] = []
                new_data[metric_name][model_name].append((timestamp, metric_data))

        # Update cached_data efficiently
        for metric_name, model_data in new_data.items():
            if metric_name not in cached_data:
                cached_data[metric_name] = model_data
            else:
                cached_data[metric_name].update(model_data)

    except sqlite3.Error as e:
        logging.error(f"SQLite error in {filepath}: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {filepath}: {e}")
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
    finally:
        if conn:
            conn.close()


async def load_data():
    tasks = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".sqlite"):
            filepath = os.path.join(DATA_DIR, filename)
            tasks.append(load_data_from_db(filepath))

    await asyncio.gather(*tasks)
    logging.info(f"Loaded data for {len(cached_data)} metrics")
    if len(cached_data) == 0:
        logging.warning(
            "No data was loaded. Check if SQLite files exist and contain recent data."
        )


async def background_data_loader():
    while True:
        await load_data()
        await asyncio.sleep(UPDATE_INTERVAL)


def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# Start the background data loader
threading.Thread(target=background_data_loader, daemon=True).start()


def create_trace(model_name, metric_name, data_points, row, col):
    return (
        go.Scattergl(
            x=[point[0] for point in data_points],
            y=[point[1] for point in data_points],
            mode="lines",
            name=f"{model_name} - {metric_name}",
        ),
        row,
        col,
    )


def create_plots(selected_model):
    global cached_data
    start_time = time.time()

    all_data = {}
    selected_models = selected_model.split(",")
    for metric, data in cached_data.items():
        all_data[metric] = {
            model: data[model] for model in selected_models if model in data
        }

    data_prep_time = time.time() - start_time
    print(f"Data preparation took {data_prep_time:.2f} seconds")

    num_metrics = len(all_data)
    if num_metrics == 0:
        logging.warning("No valid data found.")
        return None

    num_cols = 2
    num_rows = (num_metrics + num_cols - 1) // num_cols
    fig = make_subplots(
        rows=num_rows, cols=num_cols, subplot_titles=list(all_data.keys())
    )

    subplot_creation_time = time.time() - start_time - data_prep_time
    print(f"Subplot creation took {subplot_creation_time:.2f} seconds")

    now = datetime.now()
    twenty_four_hours_ago = now - timedelta(hours=24)

    trace_creation_start = time.time()

    with ThreadPoolExecutor() as executor:
        futures = []
        for index, (metric_name, model_data) in enumerate(all_data.items()):
            row = index // num_cols + 1
            col = index % num_cols + 1

            for model_name, metric_data_list in model_data.items():
                if isinstance(metric_data_list[0][1], list):
                    for label_set in metric_data_list[0][1]:
                        data_points = []
                        for timestamp, metric_data in metric_data_list:
                            if timestamp >= twenty_four_hours_ago:
                                for data_point in metric_data:
                                    if data_point["labels"] == label_set["labels"]:
                                        try:
                                            value = float(data_point["value"])
                                            data_points.append((timestamp, value))
                                        except ValueError:
                                            logging.warning(
                                                f"Invalid numeric value for {model_name} - {metric_name}: {data_point['value']}"
                                            )
                        if not data_points:
                            continue
                        data_points.sort(key=lambda x: x[0])
                        futures.append(
                            executor.submit(
                                create_trace,
                                model_name,
                                str(label_set["labels"]),
                                data_points,
                                row,
                                col,
                            )
                        )
                else:
                    data_points = []
                    for timestamp, metric_data in metric_data_list:
                        if timestamp >= twenty_four_hours_ago:
                            try:
                                value = float(metric_data)
                                data_points.append((timestamp, value))
                            except ValueError:
                                logging.warning(
                                    f"Invalid numeric value for {model_name} - {metric_name}: {metric_data}"
                                )
                    if not data_points:
                        continue
                    data_points.sort(key=lambda x: x[0])
                    futures.append(
                        executor.submit(
                            create_trace, model_name, metric_name, data_points, row, col
                        )
                    )

        for future in as_completed(futures):
            trace, row, col = future.result()
            fig.add_trace(trace, row=row, col=col)

    trace_creation_time = time.time() - trace_creation_start
    print(f"Trace creation took {trace_creation_time:.2f} seconds")

    layout_update_start = time.time()
    fig.update_layout(
        height=300 * num_rows,
        showlegend=True,
        template="plotly_dark",
        font=dict(family="Arial", size=10, color="white"),
        paper_bgcolor="rgb(30, 30, 30)",
        plot_bgcolor="rgb(30, 30, 30)",
    )
    fig.update_xaxes(title_text="Time", tickformat="%Y-%m-%d %H:%M:%S")
    fig.update_yaxes(title_text="Value")
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{y:.3f}")

    layout_update_time = time.time() - layout_update_start
    print(f"Layout update took {layout_update_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total plot creation took {total_time:.2f} seconds")

    return fig


def get_latest_rows(db_file, hours=1):
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=hours)
    one_hour_ago_timestamp = int(one_hour_ago.timestamp())

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        f"SELECT timestamp, data FROM json_data WHERE timestamp >= {one_hour_ago_timestamp} ORDER BY timestamp DESC"
    )
    rows = cursor.fetchall()

    conn.close()

    if not rows:
        print(
            f"No rows found in the last {hours} hour(s). Showing info for last 5 rows:"
        )
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT timestamp, data FROM json_data ORDER BY timestamp DESC LIMIT 5"
        )
        rows = cursor.fetchall()
        conn.close()
        for timestamp, _ in rows:
            print(f"  {datetime.fromtimestamp(timestamp)}")

    return rows


def extract_stats(data_json):
    try:
        data = json.loads(data_json)
        total_prompt_tokens = float(data["vllm:prompt_tokens_total"][0]["value"])
        total_generation_tokens = float(
            data["vllm:generation_tokens_total"][0]["value"]
        )
        total_requests = sum(
            float(item["value"]) for item in data["vllm:request_success_total"]
        )
        avg_prompt_throughput = float(
            data["vllm:avg_prompt_throughput_toks_per_s"][0]["value"]
        )
        avg_generation_throughput = float(
            data["vllm:avg_generation_throughput_toks_per_s"][0]["value"]
        )
        gpu_cache_usage_perc = float(data["vllm:gpu_cache_usage_perc"][0]["value"])
        num_requests_running = float(data["vllm:num_requests_running"][0]["value"])

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error extracting stats from data: {str(e)}")
        return None

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_generation_tokens": total_generation_tokens,
        "total_requests": total_requests,
        "avg_prompt_throughput": avg_prompt_throughput,
        "avg_generation_throughput": avg_generation_throughput,
        "gpu_cache_usage_perc": gpu_cache_usage_perc,
        "num_requests_running": num_requests_running,
    }


def get_data(db_file, hours):
    latest_rows = get_latest_rows(db_file, hours)

    if not latest_rows:
        print(f"No rows found for the last {hours} hour(s).")
        return

    print(f"Processing {len(latest_rows)} rows.")

    valid_stats = [
        extract_stats(data)
        for _, data in latest_rows
        if extract_stats(data) is not None
    ]

    if not valid_stats:
        print("No valid statistics could be extracted from the rows.")
        return

    first_stats = valid_stats[-1]  # Oldest row
    last_stats = valid_stats[0]  # Newest row

    tokens_processed = (
        last_stats["total_prompt_tokens"]
        - first_stats["total_prompt_tokens"]
        + last_stats["total_generation_tokens"]
        - first_stats["total_generation_tokens"]
    )
    requests_processed = last_stats["total_requests"] - first_stats["total_requests"]

    avg_prompt_throughput = sum(
        stat["avg_prompt_throughput"] for stat in valid_stats
    ) / len(valid_stats)
    avg_generation_throughput = sum(
        stat["avg_generation_throughput"] for stat in valid_stats
    ) / len(valid_stats)
    avg_num_requests_running = sum(
        stat["num_requests_running"] for stat in valid_stats
    ) / len(valid_stats)
    avg_gpu_cache_usage_perc = sum(
        stat["gpu_cache_usage_perc"] for stat in valid_stats
    ) / len(valid_stats)

    return (
        f"\nStats for the last {hours} hour(s):\n"
        f"Tokens processed: {tokens_processed:,.0f}\n"
        f"Requests processed: {requests_processed:,.0f}\n"
        f"Average prompt throughput: {avg_prompt_throughput:.2f} tokens/s\n"
        f"Average generation throughput: {avg_generation_throughput:.2f} tokens/s\n"
        f"Average tokens per request: {tokens_processed/requests_processed:,.2f} tokens\n"
        f"Average number of requests running: {avg_num_requests_running:.2f} requests\n"
        f"Average GPU cache usage percent: {avg_gpu_cache_usage_perc * 100:.2f}%"
    )


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    model_names = [
        name[:-7]
        for name in os.listdir(DATA_DIR)
        if name.endswith(".sqlite") and os.path.isfile(os.path.join(DATA_DIR, name))
    ]
    valid_model_names = set(model_names)

    if request.method == "POST":
        selected_model = request.form.get("model_select")
    else:
        selected_model = model_names[0] if model_names else None

    plot_div = None
    error_message = None
    if selected_model in valid_model_names:
        try:
            fig = create_plots(selected_model)
            if fig is not None:
                fig.update_layout(showlegend=False)
                plot_div = pyo.plot(fig, output_type="div", include_plotlyjs=True)
            else:
                error_message = "No data available for the selected model."
        except Exception as e:
            logging.error(f"Error creating plot: {str(e)}")
            error_message = (
                "An error occurred while creating the plot. Please try again later."
            )

        result = get_data(f"{DATA_DIR}/{selected_model}.sqlite", 24)
    else:
        logging.error(f"Invalid model selected: {selected_model}")
        result = None

    return render_template(
        "index.html",
        plot_div=plot_div,
        model_name=selected_model,
        model_names=model_names,
        result=result,
        error_message=error_message,
    )


@app.route("/favicon.ico")
def favicon():
    return send_file("favicon.ico", mimetype="image/vnd.microsoft.icon")


if __name__ == "__main__":
    import uvicorn
    from asgiref.wsgi import WsgiToAsgi

    # Initial data load
    logging.info("Starting initial data load")
    asyncio.run(load_data())
    logging.info("Initial data load complete")

    # Create a new event loop for the background task
    loop = asyncio.new_event_loop()

    def start_background_loop():
        asyncio.set_event_loop(loop)
        loop.create_task(background_data_loader())
        loop.run_forever()

    t = threading.Thread(target=start_background_loop, daemon=True)
    t.start()

    asgi_app = WsgiToAsgi(app)
    uvicorn.run(asgi_app, host=HOST, port=PORT)
