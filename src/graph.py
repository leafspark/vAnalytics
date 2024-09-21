import asyncio
import json
import logging
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file
from functools import lru_cache
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
from scipy.interpolate import make_interp_spline

# Set up logging with a higher level
logging.basicConfig(
    level=logging.WARNING,  # Changed from DEBUG to WARNING
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
    data_dir = "./data"

    tasks = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".sqlite"):
            filepath = os.path.join(data_dir, filename)
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
        await asyncio.sleep(30)  # Check for updates every 30 seconds


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


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    data_dir = "./data"
    model_names = [
        name[:-7]
        for name in os.listdir(data_dir)
        if name.endswith(".sqlite") and os.path.isfile(os.path.join(data_dir, name))
    ]

    if request.method == "POST":
        selected_model = request.form.get("model_select")
    else:
        selected_model = model_names[0] if model_names else None

    plot_div = None
    error_message = None
    if selected_model:
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

        command = [
            "python",
            "get_data.py",
            "--hours",
            "24",
            f".\\data\\{selected_model}.sqlite",
        ]

        result = subprocess.run(command, capture_output=True, text=True)
    else:
        result = None

    return render_template(
        "index.html",
        plot_div=plot_div,
        model_name=selected_model,
        model_names=model_names,
        result=result.stdout if result else None,
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
    uvicorn.run(asgi_app, host="0.0.0.0", port=4421)
