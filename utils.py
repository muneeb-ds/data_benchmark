import argparse
import os, psutil
from time import perf_counter
import shutil
import pandas as pd
import numpy as np

class PerformanceTracker:
    def __init__(self) -> None:
        self.performance_df = pd.DataFrame()
        self.performance_dict = {}
        self.performance_dict['pandas'] = {}
        self.performance_dict['modin'] = {}
        self.performance_dict['polars'] = {}

    def add(self, stats, framework):
        self.performance_dict[framework]['memory_usage (MB)'] = stats[0]
        self.performance_dict[framework]['time_consumed (s)'] = stats[1]
        return self
    
    def combine_stats(self, stats_list, operation):
        self.performance_dict["operation"] = operation
        self.add(stats_list[0], "pandas")
        self.add(stats_list[1], "modin")
        self.add(stats_list[2], "polars")
        temp_df = pd.DataFrame(self.performance_dict)
        self.performance_df = pd.concat([self.performance_df, temp_df], axis = 0)
        return self

def argument_parser():
    parser = argparse.ArgumentParser(description="benchmark arguments")
    parser.add_argument("--data_path",
                        required=True,
                        help = "path where csv to be read is saved")
    parser.add_argument("--rows",
                        type= int,
                        default=1000,
                        help = "number of rows of created dataframe")
    parser.add_argument("--columns",
                        type= int,
                        default=1000,
                        help = "number of columns of created dataframe")
    parser.add_argument("--iterations",
                        type= int,
                        default=1,
                        help = "number of iterations to run for all operations")
    args = parser.parse_args()

    return args

def profile(func):
    def elapsed_since(start):
        return perf_counter() - start

    def get_mem_usage():
        process = psutil.Process(os.getpid())
        memory_pct = process.memory_full_info().rss / (1024 * 1024)
        return(memory_pct)

    def wrapper(*args, **kwargs):
        
        mem_before = get_mem_usage()
        start = perf_counter()
        result = func(*args, **kwargs)
        mem_after = round(get_mem_usage() - mem_before, 2)
        elapsed_time = round(elapsed_since(start), 4)
        return result, (mem_after, elapsed_time)
    return wrapper

def create_dataframe_dict(rows, columns):
    dataframe = {}
    dataframe['age'] = np.random.randint(0,100, rows)
    dataframe['gender'] = np.random.choice(['male', 'female', np.NaN], rows)
    dataframe['height'] = np.random.randint(100,200, rows)
    dataframe['weight'] = np.random.uniform(30.0, 100.0, rows)
    dataframe['hobby'] = np.random.choice(["games", "books", "movies", "programming", np.NaN], rows)
    
    random_cols = max(columns - 5,5)
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
    perf_df_mean = perf_df.groupby(['stat','operation','framework'], as_index = False)['values'].mean()
    perf_df_mean['mean_stats'] = np.round(perf_df_mean['values'],4)
    perf_df_mean.drop(columns = ['values'], inplace = True)

    return perf_df_mean.pivot(index = 'operation', columns = ['stat','framework'], values = 'mean_stats')