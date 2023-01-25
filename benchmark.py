import shutil
from utils import *
from operations import *
import ray
ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, include_dashboard=False)

args = argument_parser()

DATA_PATH = args.data_path
ROW_SIZE = args.rows
COLUMN_SIZE = args.columns
ITERS = args.iterations

def track_operations_perf(i):

    perf_track = PerformanceTracker()

    df_pd, stats_pd = pd_read_csv(DATA_PATH)
    df_md, stats_md = md_read_csv(DATA_PATH)
    df_pl, stats_pl = pl_read_csv(DATA_PATH)
    operation = f"reading csv of shape:{df_pd.shape}"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    rand_arr = np.random.randint(0,100,df_pd.shape[0])

    df_pd, stats_pd = pd_md_add_column(df_pd, rand_arr)
    df_md, stats_md = pd_md_add_column(df_pd, rand_arr)
    df_pl, stats_pl = pl_add_column(df_pl, rand_arr)
    operation = "add column"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    _, stats_pd= pd_md_get_date_range(df_pd)
    _, stats_pd= pd_md_get_date_range(df_md)
    _, stats_pd= pl_get_date_range(df_pl)
    operation = "get date range"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)
                                
    pd_float_cols = [col for col in df_pd.columns if df_pd[col].dtype in ['float']]
    md_float_cols = [col for col in df_md.columns if df_md[col].dtype in ['float']]
    pl_float_cols = [col for col,dtype in df_pl.schema.items() if str(dtype) in ['Float64', 'Float16']]

    filter_col = np.random.choice(pd_float_cols)

    filter_val, stats_pd = col_mean(df_pd, filter_col)
    filter_val, stats_md = col_mean(df_md, filter_col)
    filter_val, stats_pl = col_mean(df_pl, filter_col)
    operation = "get column mean val"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    pd_filter, stats_pd = pd_md_filter_values(df_pd, filter_col, filter_val)
    md_filter, stats_md = pd_md_filter_values(df_md, filter_col, filter_val)
    pl_filter, stats_pl = pl_filter_values(df_pl, filter_col, filter_val)
    operation = "filter values based on col mean"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    df_str_cols = [col for col in df_pd.columns if df_pd[col].dtype in ['object', 'str']]
    md_str_cols = [col for col in df_md.columns if df_md[col].dtype in ['object','str']]
    pl_str_cols = [col for col,dtype in df_pl.schema.items() if str(dtype) in ['str']]

    groupby_col = np.random.choice(df_str_cols)

    grouped_pd, stats_pd = pd_md_groupby(df_pd, groupby_col, filter_col)
    grouped_md, stats_md = pd_md_groupby(df_md, groupby_col, filter_col)
    grouped_pl, stats_pl = pl_groupby(df_pl, groupby_col, filter_col)
    operation = "groupby aggregation (sum, mean, std)"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    pd_merged, stats_pd = pd_md_merge(df_pd, grouped_pd, on = groupby_col)
    md_merged, stats_md = pd_md_merge(df_md, grouped_md, on = groupby_col)
    pl_merged, stats_pl = pl_merge(df_pl, grouped_pl, on = groupby_col)
    operation = "merging grouped col to original df"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    df_pd, stats_pd = pd_md_combined_groupby_merge(df_pd, groupby_col, filter_col)
    df_md, stats_md = pd_md_combined_groupby_merge(df_md, groupby_col, filter_col)
    df_pl, stats_pl = pl_combined_groupby_merge(df_pl, groupby_col, filter_col)
    operation = "combined groupby merge"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    pd_concated, stats_pd = pd_concat(pd_merged, pd_filter)
    md_concated, stats_md = md_concat(md_merged, md_filter)
    pl_concated, stats_pl = pl_concat(pl_merged, pl_filter)
    operation = "horizontal concatenatenation"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    _, stats_pd = pd_md_fill_nulls(pd_concated)
    _, stats_md = pd_md_fill_nulls(md_concated)
    _, stats_pl = pl_fill_nulls(pl_concated)
    operation = "fill nulls with 0"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    _, stats_pd = pd_md_drop_nulls(pd_concated)
    _, stats_md = pd_md_drop_nulls(md_concated)
    _, stats_pl = pl_drop_nulls(pl_concated)
    operation = "drop nulls"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    df_dict = create_dataframe_dict(ROW_SIZE, COLUMN_SIZE)

    pd_new_df, stats_pd = pd_create_df(df_dict)
    md_new_df, stats_md = md_create_df(df_dict)
    pl_new_df, stats_pl = pl_create_df(df_dict)
    operation = f"create dataframe of size: ({ROW_SIZE},{COLUMN_SIZE})"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    _, stats_pd = describe_df(pd_new_df)
    _, stats_md = describe_df(md_new_df)
    _, stats_pl = describe_df(pl_new_df)
    operation = "describe stats of df"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    _, stats_pd = pd_md_save_to_csv(pd_new_df)
    _, stats_md = pd_md_save_to_csv(md_new_df)
    _, stats_pl = pl_save_to_csv(pl_new_df)
    operation = "save to csv"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    parquet_path = "sample_data.parquet"

    remove_parquets(parquet_path)
    _, stats_pd = pd_md_save_to_parquet(pd_new_df)
    remove_parquets(parquet_path)
    _, stats_md = pd_md_save_to_parquet(md_new_df)
    remove_parquets(parquet_path)
    _, stats_pl = pl_save_to_parquet(pl_new_df)
    operation = "save to parquet"

    perf_track.combine_stats([stats_pd, stats_md, stats_pl], operation)

    performance_df = perf_track.performance_df.reset_index(names = "stat")
    performance_df = pd.melt(performance_df, id_vars=['stat','operation'], var_name = 'framework',value_name="values")
    performance_df['iteration'] = i
    return performance_df

perf_df = pd.DataFrame()
for i in range(ITERS):
    temp_df = track_operations_perf(i)
    perf_df = pd.concat([perf_df, temp_df], axis = 0)

perf_df = format_perf_df(perf_df)
perf_df.to_csv("performance_benchmarks.csv")