import pandas as pd
import csv
import numpy as np
import pickle

def write_history_to_csv(history, csv_filename="history_data.csv", chunk_size=100000, use_pandas=False):
    """
    Writes history data to a CSV file in an incremental fashion to avoid high memory usage.

    Args:
    - history (dict or list): Nested structure containing frames with state, rewards, and info.
    - csv_filename (str): Name of the CSV file to write data to.
    - chunk_size (int): Number of rows to store before writing to disk (used when use_pandas=True).
    - use_pandas (bool): Whether to use Pandas' `to_csv` for writing in chunks.
    """
    headers = ["regime_idx", "frame_idx", "a1x", "a1y", "a2x", "a2y", "reward_loc", "r1", "r2", 
               "activated", "collected", "terminated", "steps_without_reward"]

    if use_pandas:
        # Pandas Chunk-Based Writing
        with open(csv_filename, mode="w", newline="") as file:
            chunk = []
            for regime_idx, frames in enumerate(history):
                for frame_idx, frame in frames.items():
                    ((a1x, a1y), (a2x, a2y)), reward_loc = frame["state"]
                    r1, r2 = frame["rewards"]
                    info = frame["info"]

                    chunk.append({
                        "regime_idx": regime_idx,
                        "frame_idx": frame_idx,
                        "a1x": a1x, "a1y": a1y,
                        "a2x": a2x, "a2y": a2y,
                        "reward_loc": reward_loc,
                        "r1": r1, "r2": r2,
                        "activated": info["activated"],
                        "collected": info["collected"],
                        "terminated": info["terminated"],
                        "steps_without_reward": info["steps_without_reward"]
                    })

                    if len(chunk) >= chunk_size:
                        df = pd.DataFrame(chunk)
                        df.to_csv(file, mode="a", header=file.tell()==0, index=False)
                        chunk = []

            # Write remaining chunk
            if chunk:
                df = pd.DataFrame(chunk)
                df.to_csv(file, mode="a", header=file.tell()==0, index=False)

    else:
        # Standard CSV Writer (Row-by-Row)
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()

            for regime_idx, frames in enumerate(history):
                for frame_idx, frame in frames.items():
                    ((a1x, a1y), (a2x, a2y)), reward_loc = frame["state"]
                    r1, r2 = frame["rewards"]
                    info = frame["info"]

                    writer.writerow({
                        "regime_idx": regime_idx,
                        "frame_idx": frame_idx,
                        "a1x": a1x, "a1y": a1y,
                        "a2x": a2x, "a2y": a2y,
                        "reward_loc": reward_loc,
                        "r1": r1, "r2": r2,
                        "activated": info["activated"],
                        "collected": info["collected"],
                        "terminated": info["terminated"],
                        "steps_without_reward": info["steps_without_reward"]
                    })
