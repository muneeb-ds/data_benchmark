import argparse
import os, psutil
import tracemalloc
from memory_profiler import memory_usage
from time import perf_counter
import shutil
import pandas as pd
import numpy as np


def argument_parser():
    parser = argparse.ArgumentParser(description="benchmark arguments")
    parser.add_argument("--data_path", required=True, help="path where csv to be read is saved")
    parser.add_argument(
        "--rows", type=int, default=1000, help="number of rows of created dataframe"
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=1000,
        help="number of columns of created dataframe",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="number of iterations to run for all operations",
    )
    parser.add_argument("--save_dir", default="", help="directory to save output benchmark csv")
    parser.add_argument(
        "--frameworks",
        default=["pandas", "modin", "polars"],
        nargs="*",
        help="frameworks to benchmark on (pandas, modin, polars, duckdb supported)",
        choices=["pandas", "modin", "polars", "duckdb"],
    )
    args = parser.parse_args()

    return args


def profile(func):
    def elapsed_since(start):
        return perf_counter() - start

    def get_mem_usage():
        process = psutil.Process(os.getpid())
        memory_pct = process.memory_full_info().rss / (1024 * 1024)
        return memory_pct

    def wrapper(*args, **kwargs):

        start = perf_counter()
        mem_before = get_mem_usage()
        result = func(*args, **kwargs)
        # mem_after, result = memory_usage((func, args, kwargs), retval=True, max_usage=True)
        mem_after = get_mem_usage()
        mem_after = round(mem_after - mem_before, 2)
        elapsed_time = round(elapsed_since(start), 4)
        return result, (mem_after, elapsed_time)

    return wrapper


def create_dataframe_dict(rows, columns):
    dataframe = {}
    dataframe["age"] = np.random.randint(0, 100, rows)
    dataframe["gender"] = np.random.choice(["male", "female", np.NaN], rows)
    dataframe["height"] = np.random.randint(100, 200, rows)
    dataframe["weight"] = np.random.uniform(30.0, 100.0, rows)
    dataframe["hobby"] = np.random.choice(["games", "books", "movies", "programming", np.NaN], rows)

    random_cols = max(columns - 5, 5)
    for col in range(random_cols):
        dataframe[f"randomCol_{col}"] = np.random.rand(rows)

    return dataframe


def remove_parquets(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError:
        return


def format_perf_df(perf_df):
    perf_df_mean = perf_df.groupby(["operation", "framework", "stat"], as_index=False)[
        "values"
    ].mean()
    perf_df_mean["mean_stats"] = np.round(perf_df_mean["values"], 4)
    perf_df_mean.drop(columns=["values"], inplace=True)

    return perf_df_mean.pivot(index=["operation", "stat"], columns="framework", values="mean_stats")
