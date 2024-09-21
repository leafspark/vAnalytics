import os
import json
import argparse
import sqlite3
from datetime import datetime, timedelta


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


def main(db_file, hours):
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

    print(f"\nStats for the last {hours} hour(s):")
    print(f"Tokens processed: {tokens_processed:,.0f}")
    print(f"Requests processed: {requests_processed:,.0f}")
    print(f"Average prompt throughput: {avg_prompt_throughput:.2f} tokens/s")
    print(f"Average generation throughput: {avg_generation_throughput:.2f} tokens/s")
    print(
        f"Average number of requests running: {avg_num_requests_running:.2f} requests"
    )
    print(f"Average GPU cache usage percent: {avg_gpu_cache_usage_perc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract stats from a SQLite database for a specified time period"
    )
    parser.add_argument("db_file", help="Path to the SQLite database file")
    parser.add_argument(
        "--hours", type=int, default=1, help="Number of hours to look back (default: 1)"
    )
    args = parser.parse_args()

    main(args.db_file, args.hours)
