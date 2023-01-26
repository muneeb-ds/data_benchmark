from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import modin.pandas as md
import polars as pl
import ray

from utils import profile, create_dataframe_dict, remove_parquets


class PerformanceTracker(ABC):
    def __init__(self, args) -> None:
        self.performance_dict = {}
        self.data_path = args.data_path
        self.row_size = args.rows
        self.column_size = args.columns
        self.iters = args.iterations

    def add(self, stats, operation):
        self.performance_dict[operation] = {}
        self.performance_dict[operation]["memory_usage (MB)"] = stats[0]
        self.performance_dict[operation]["time_consumed (S)"] = stats[1]
        return self

    def get_stats_df(self):
        self.performance_df = pd.DataFrame.from_dict(self.performance_dict, orient="index")
        return self.performance_df

    @abstractmethod
    def read_csv(self, path):
        pass

    @abstractmethod
    def add_column(self, df, array):
        pass

    @abstractmethod
    def get_date_range(self):
        pass

    @profile
    def col_mean(self, df, filter_col):
        filter_val = df[filter_col].mean()
        return filter_val

    @abstractmethod
    def filter_vals(self, df, filter_col, filter_val):
        pass

    @abstractmethod
    def groupby(self, df, groupby_col, agg_col):
        pass

    @abstractmethod
    def merge(self, left, right, on):
        pass

    @abstractmethod
    def groupby_merge(self, df, groupby_col, agg_col):
        pass

    @abstractmethod
    def concat(self, df_1, df_2):
        pass

    @abstractmethod
    def fill_na(self, df):
        pass

    @abstractmethod
    def drop_na(self, df):
        pass

    @abstractmethod
    def create_df(self, df_dict):
        pass

    @profile
    def describe_df(self, df):
        return df.describe()

    @abstractmethod
    def save_to_csv(self, df):
        pass

    @abstractmethod
    def save_to_parquet(self, df):
        pass

    def run_operations(self):
        df, stats = self.read_csv(self.data_path)
        operation = f"reading csv of shape:{df.shape}"

        self.add(stats, operation)

        rand_arr = np.random.randint(0, 100, df.shape[0])

        df, stats = self.add_column(df, rand_arr)
        operation = "add column"
        self.add(stats, operation)

        _, stats = self.get_date_range()
        operation = "get date range"
        self.add(stats, operation)

        float_cols = [col for col in df.columns if str(df[col].dtype) in ["float", "Float64", "Float16", "float64"]]

        filter_col = np.random.choice(float_cols)

        filter_val, stats = self.col_mean(df, filter_col)
        operation = "get column mean val"
        self.add(stats, operation)

        filtered_df, stats = self.filter_vals(df, filter_col, filter_val)
        operation = "filter values based on col mean"
        self.add(stats, operation)

        df_str_cols = [col for col in df.columns if str(df[col].dtype) in ["object", "str", "Utf8"]]
        groupby_col = np.random.choice(df_str_cols)

        grouped_df, stats = self.groupby(df, groupby_col, filter_col)
        operation = "groupby aggregation (sum, mean, std)"
        self.add(stats, operation)

        merged_df, stats = self.merge(df, grouped_df, groupby_col)
        operation = "merging grouped col to original df"
        self.add(stats, operation)

        merged_df, stats = self.groupby_merge(df, groupby_col, filter_col)
        operation = "combined groupby merge"
        self.add(stats, operation)

        concat_df, stats = self.concat(merged_df, filtered_df)
        operation = "horizontal concatenatenation"
        self.add(stats, operation)

        _, stats = self.fill_na(concat_df)
        operation = "fill nulls with 0"
        self.add(stats, operation)

        _, stats = self.drop_na(concat_df)
        operation = "drop nulls"
        self.add(stats, operation)

        df_dict = create_dataframe_dict(self.row_size, self.column_size)

        new_df, stats = self.create_df(df_dict)
        operation = f"create dataframe of size: ({self.row_size},{self.column_size})"
        self.add(stats, operation)

        _, stats = self.describe_df(new_df)
        operation = "describe stats of df"
        self.add(stats, operation)

        _, stats = self.save_to_csv(new_df)
        operation = "save to csv"
        self.add(stats, operation)

        parquet_path = "sample_data.parquet"
        remove_parquets(parquet_path)

        _, stats = self.save_to_parquet(new_df)
        operation = "save_to_parquet"
        self.add(stats, operation)

        perf_df = self.get_stats_df()

        return perf_df


class PandasBench(PerformanceTracker):
    ## READ CSV
    @profile
    def read_csv(self, path):
        df = pd.read_csv(path)
        return df

    @profile
    def add_column(self, df, array):
        df["rand_nums"] = array
        return df

    @profile
    def get_date_range(self):
        pd_dates = pd.date_range(start="1990-01-01", end="2050-12-31")
        return pd_dates

    @profile
    def filter_vals(self, df, filter_col, filter_val):
        return df.loc[df[filter_col] > filter_val, :]

    @profile
    def groupby(self, df, groupby_col, agg_col):
        return df.groupby([groupby_col], as_index=False).agg(
            agg_mean=(f"{agg_col}", "mean"), agg_sum=(f"{agg_col}", "sum"), agg_std=(f"{agg_col}", "std")
        )

    @profile
    def merge(self, left, right, on):
        return pd.merge(left, right, on=[on], how="left")

    @profile
    def groupby_merge(self, df, groupby_col, agg_col):
        grouped = df.groupby([groupby_col], as_index=False).agg(
            agg_mean=(f"{agg_col}", "mean"), agg_sum=(f"{agg_col}", "sum"), agg_std=(f"{agg_col}", "std")
        )
        return pd.merge(df, grouped, on=[groupby_col], how="left")

    @profile
    def concat(self, df_1, df_2):
        return pd.concat([df_1, df_2], axis=0)

    @profile
    def fill_na(self, df):
        df.fillna(0)

    @profile
    def drop_na(self, df):
        df.dropna()

    @profile
    def create_df(self, df_dict):
        return pd.DataFrame(df_dict)

    @profile
    def save_to_csv(self, df):
        df.to_csv("sample_data.csv", index=False)

    @profile
    def save_to_parquet(self, df):
        df.to_parquet("sample_data.parquet", index=False)

    def get_stats_df(self):
        stats_df = super().get_stats_df()
        stats_df["framework"] = "pandas"
        return stats_df


class ModinBench(PandasBench, PerformanceTracker):
    def __init__(self, args):
        super().__init__(args)
        ray.init(runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}}, include_dashboard=False)

    @profile
    def create_df(self, df_dict):
        return md.DataFrame(df_dict)

    @profile
    def read_csv(self, path):
        df = md.read_csv(path)
        return df

    @profile
    def get_date_range(self):
        pd_dates = md.date_range(start="1990-01-01", end="2050-12-31")
        return pd_dates

    @profile
    def merge(self, left, right, on):
        return md.merge(left, right, on=[on], how="left")

    @profile
    def groupby_merge(self, df, groupby_col, agg_col):
        grouped = df.groupby([groupby_col], as_index=False).agg(
            agg_mean=(f"{agg_col}", "mean"), agg_sum=(f"{agg_col}", "sum"), agg_std=(f"{agg_col}", "std")
        )
        return md.merge(df, grouped, on=[groupby_col], how="left")

    @profile
    def concat(self, df_1, df_2):
        return md.concat([df_1, df_2], axis=0)

    def get_stats_df(self):
        stats_df = super().get_stats_df()
        stats_df["framework"] = "modin"
        return stats_df


class PolarsBench(PerformanceTracker):
    @profile
    def read_csv(self, path):
        df = pl.read_csv(path)
        return df

    @profile
    def add_column(self, df, array):
        df = df.with_columns([pl.Series(array).alias("rand_num")])
        return df

    @profile
    def get_date_range(self):
        pl_dates = pl.date_range(low=datetime(1990, 1, 1), high=datetime(2050, 12, 31), interval="1d", closed="left")
        return pl_dates

    @profile
    def filter_vals(self, df, filter_col, filter_val):
        return df.filter(pl.col(filter_col) > filter_val)

    @profile
    def groupby(self, df, groupby_col, agg_col):
        q = (
            df.lazy()
            .groupby(groupby_col)
            .agg(
                [
                    pl.col(agg_col).sum().suffix("_sum"),
                    pl.col(agg_col).mean().suffix("_mean"),
                    pl.col(agg_col).std().suffix("_std"),
                ]
            )
        )
        pl_groupby = q.collect()
        return pl_groupby

    @profile
    def merge(self, left, right, on):
        return left.join(right, on=[on], how="left")

    @profile
    def groupby_merge(self, df, groupby_col, agg_col):
        return df.select(
            [
                pl.all(),
                pl.col(agg_col).sum().over(groupby_col).alias(f"{agg_col}_sum"),
                pl.col(agg_col).mean().over(groupby_col).alias(f"{agg_col}_mean"),
            ]
        )

    @profile
    def concat(self, df_1, df_2):
        return pl.concat([df_1, df_2], how="diagonal")

    @profile
    def fill_na(self, df):
        df.fill_null(0)

    @profile
    def drop_na(self, df):
        df.drop_nulls()

    @profile
    def create_df(self, df_dict):
        return pl.from_dict(df_dict)

    @profile
    def save_to_csv(self, df):
        df.write_csv("sample_data.csv")

    @profile
    def save_to_parquet(self, df):
        df.write_parquet("sample_data.parquet")

    def get_stats_df(self):
        stats_df = super().get_stats_df()
        stats_df["framework"] = "polars"
        return stats_df
