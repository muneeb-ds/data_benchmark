from utils import argument_parser, format_perf_df
import pandas as pd
from operations import PandasBench, ModinBench, PolarsBench
import ray

ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, include_dashboard=False)

args = argument_parser()

ITERS = args.iterations

frameworks = [PandasBench(args), ModinBench(args), PolarsBench(args)]

perf_df = pd.DataFrame()
for iter in range(ITERS):
    df_stats = [framework.run_operations() for framework in frameworks]

    df_perf_stats = pd.concat(df_stats, axis = 0)
    performance_df = df_perf_stats.reset_index(names = "operation")
    performance_df = pd.melt(performance_df, id_vars=['operation', 'framework'], var_name = 'stat',value_name="values")
    performance_df['iteration'] = iter
    perf_df = pd.concat([perf_df, performance_df], axis = 0)

perf_df = format_perf_df(perf_df)
perf_df.to_csv("performance_benchmarks.csv")