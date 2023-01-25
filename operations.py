from abc import ABC, abstractmethod
from collections import defaultdict
import pandas as pd
import modin.pandas as md
import polars as pl
import numpy as np
from time import perf_counter
import os, psutil


class PerformanceTracker:
    def __init__(self) -> None:
        self.performance_dict = defaultdict()

    def add(self, stats, framework, operation):
        self.performance_dict['operation'] = operation
        self.performance_dict[framework]['memory_usage'] = stats[0]
        self.performance_dict[framework]['time_consumed'] = stats[1]
        return self
        

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

@profile
def pd_read_csv(path):
    df = pd.read_csv(path)
    return df

@profile
def md_read_csv(path):
    df = md.read_csv(path)
    return df

@profile
def pl_read_csv(path):
    df = pl.read_csv(path)
    return df

