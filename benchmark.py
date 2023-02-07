import gc
import logging
import os
import pandas as pd

from operations import PerformanceTracker
from utils import argument_parser, format_perf_df

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = argument_parser()
    ITERS = args.iterations
    perf_df = pd.DataFrame()

    frameworks = [
        cls(args)
        for cls in PerformanceTracker.__subclasses__()
        if cls.__name__.split("Bench")[0].lower() in args.frameworks
    ]
    logger.critical("Starting Benchmarking...")

    for i in range(ITERS):
        logger.critical("Running iteration: %s", i + 1)
        df_stats = [framework.run() for framework in frameworks]
        df_perf_stats = pd.concat(df_stats, axis=0)
        performance_df = df_perf_stats.reset_index(names="operation")
        performance_df = pd.melt(
            performance_df, id_vars=["operation", "framework"], var_name="stat", value_name="values"
        )
        performance_df["iteration"] = i
        perf_df = pd.concat([perf_df, performance_df], axis=0)
        del df_stats
        gc.collect()

    perf_df = format_perf_df(perf_df)
    perf_df.to_csv(os.path.join(args.save_dir, "performance_benchmarks.csv"))
