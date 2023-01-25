from datetime import datetime, timedelta
import pandas as pd
import modin.pandas as md
import polars as pl

from utils import profile


## READ CSV
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

## ADD COLUMN
@profile
def pd_md_add_column(df, array):
    df['rand_nums'] = array
    return df

@profile
def pl_add_column(df, array):
    df = df.with_columns([pl.Series(array).alias("rand_num")])
    return df

## Date ranges
@profile
def pd_md_get_date_range(df):
    pd_dates = pd.date_range(start = "1990-01-01", end = "2050-12-31")
    # df['date'] = pd_dates
    return pd_dates

@profile
def pl_get_date_range(df):
    pl_dates = pl.date_range(low = datetime(1990, 1,1), high = datetime(2050,12,31), interval = '1d', closed = "left")
    # df = df.with_columns([pl_dates.alias("date")])
    return pl_dates

@profile
def col_mean(df, filter_col):
    filter_val = df[filter_col].mean()
    return filter_val

@profile
def pd_md_filter_values(df,filter_col, filter_val):
    return df.loc[df[filter_col]>filter_val,:]

@profile
def pl_filter_values(df, filter_col, filter_val):
    return df.filter(pl.col(filter_col)>filter_val)

@profile
def pd_md_groupby(df, groupby_col, agg_col):
    return df.groupby([groupby_col], as_index = False).agg(agg_mean=(f'{agg_col}','mean'),
                                                              agg_sum=(f'{agg_col}','sum'),
                                                               agg_std=(f'{agg_col}','std'))

@profile
def pl_groupby(df, groupby_col, agg_col):
    q = (df.lazy().groupby(groupby_col).agg([
        pl.col(agg_col).sum().suffix("_sum"),
        pl.col(agg_col).mean().suffix("_mean"),
        pl.col(agg_col).std().suffix("_std")
    ]))
    pl_groupby = q.collect()
    return pl_groupby

@profile
def pd_md_merge(left, right, on):
    return pd.merge(left, right, on = [on], how = 'left')

@profile
def pl_merge(left, right, on):
    return left.join(right, on =[on], how = 'left')

@profile
def pd_md_combined_groupby_merge(df, groupby_col, agg_col):
    grouped = df.groupby([groupby_col], as_index = False).agg(agg_mean=(f'{agg_col}','mean'),
                                                              agg_sum=(f'{agg_col}','sum'),
                                                               agg_std=(f'{agg_col}','std'))
    return pd.merge(df, grouped, on = [groupby_col], how = 'left')

@profile
def pl_combined_groupby_merge(df, groupby_col, agg_col):
    return df.select([
    pl.all(),
    pl.col(agg_col).sum().over(groupby_col).alias(f"{agg_col}_sum"),
    pl.col(agg_col).mean().over(groupby_col).alias(f"{agg_col}_mean")
    ])

@profile
def pd_concat(df_1, df_2):
    return pd.concat([df_1,df_2], axis = 0)

@profile
def md_concat(df_1, df_2):
    return md.concat([df_1,df_2], axis = 0)

@profile
def pl_concat(df_1, df_2):
    return pl.concat([df_1,df_2], how = 'diagonal')

@profile
def pd_md_fill_nulls(df):
    df.fillna(0)

@profile
def pl_fill_nulls(df):
    df.fill_null(0)

@profile
def pd_md_drop_nulls(df):
    df.dropna()

@profile
def pl_drop_nulls(df):
    df.drop_nulls()

@profile
def pd_create_df(df_dict):
    return pd.DataFrame(df_dict)

@profile
def md_create_df(df_dict):
    return md.DataFrame(df_dict)

@profile
def pl_create_df(df_dict):
    return pl.from_dict(df_dict)

@profile
def describe_df(df):
    return df.describe()

@profile
def pd_md_save_to_csv(df):
    df.to_csv("sample_data.csv", index = False)

@profile
def pl_save_to_csv(df):
    df.write_csv("sample_data.csv")

@profile
def pd_md_save_to_parquet(df):
    df.to_parquet("sample_data.parquet", index = False)

@profile
def pl_save_to_parquet(df):
    df.write_parquet("sample_data.parquet")
