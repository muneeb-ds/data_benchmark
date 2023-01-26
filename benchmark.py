import pandas as pd
import os

from operations import PerformanceTracker
from utils import argument_parser, format_perf_df

args = argument_parser()
ITERS = args.iterations

frameworks = [
    cls(args)
    for cls in PerformanceTracker.__subclasses__()
    if cls.__name__.split("Bench")[0].lower() in args.frameworks
]

perf_df = pd.DataFrame()
for iter in range(ITERS):
    df_stats = [framework.run_operations() for framework in frameworks]

    df_perf_stats = pd.concat(df_stats, axis=0)
    performance_df = df_perf_stats.reset_index(names="operation")
    performance_df = pd.melt(performance_df, id_vars=["operation", "framework"], var_name="stat", value_name="values")
    performance_df["iteration"] = iter
    perf_df = pd.concat([perf_df, performance_df], axis=0)

perf_df = format_perf_df(perf_df)
perf_df.to_csv(os.path.join(args.save_dir,"performance_benchmarks.csv"))
