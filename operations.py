from abc import ABC, abstractmethod
from datetime import datetime
import gc
import time
import numpy as np
import pandas as pd
import logging

from utils import profile, create_dataframe_dict, remove_parquets

logger = logging.getLogger(__name__)


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
        t0 = time.perf_counter()

        df, stats = self.read_csv(self.data_path)
        operation = f"reading csv of shape:{df.shape}"
        logger.critical(f"{self.__class__.__name__}: {operation}")

        self.add(stats, operation)

        rand_arr = np.random.randint(0, 100, df.shape[0])

        operation = "add column"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        df, stats = self.add_column(df, rand_arr)
        self.add(stats, operation)

        operation = "get date range"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        _, stats = self.get_date_range()
        self.add(stats, operation)

        float_cols = [col for col in df.columns if str(df[col].dtype) in ["float", "Float64", "Float16", "float64"]]

        filter_col = np.random.choice(float_cols)

        operation = "get column mean val"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        filter_val, stats = self.col_mean(df, filter_col)
        self.add(stats, operation)

        operation = "filter values based on col mean"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        filtered_df, stats = self.filter_vals(df, filter_col, filter_val)
        self.add(stats, operation)

        df_str_cols = [col for col in df.columns if str(df[col].dtype) in ["object", "str", "Utf8"]]
        groupby_col = np.random.choice(df_str_cols)

        operation = "groupby aggregation (sum, mean, std)"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        grouped_df, stats = self.groupby(df, groupby_col, filter_col)
        self.add(stats, operation)

        operation = "merging grouped col to original df"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        merged_df, stats = self.merge(df, grouped_df, groupby_col)
        self.add(stats, operation)

        operation = "combined groupby merge"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        merged_df, stats = self.groupby_merge(df, groupby_col, filter_col)
        self.add(stats, operation)

        operation = "horizontal concatenatenation"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        concat_df, stats = self.concat(merged_df, filtered_df)
        self.add(stats, operation)

        operation = "fill nulls with 0"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        _, stats = self.fill_na(concat_df)
        self.add(stats, operation)

        operation = "drop nulls"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        _, stats = self.drop_na(concat_df)
        self.add(stats, operation)

        df_dict = create_dataframe_dict(self.row_size, self.column_size)

        operation = f"create dataframe of size: ({self.row_size},{self.column_size})"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        new_df, stats = self.create_df(df_dict)
        self.add(stats, operation)

        operation = "describe stats of df"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        _, stats = self.describe_df(new_df)
        self.add(stats, operation)

        operation = "save to csv"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        _, stats = self.save_to_csv(new_df)
        self.add(stats, operation)

        parquet_path = "sample_data.parquet"
        remove_parquets(parquet_path)

        operation = "save_to_parquet"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        _, stats = self.save_to_parquet(new_df)
        self.add(stats, operation)

        t_final = time.perf_counter() - t0
        operation = "Total stats"
        logger.critical(f"{self.__class__.__name__}: {operation}")
        self.add((np.NaN, t_final), operation)

        logger.critical(f"{self.__class__.__name__}: combining stats")
        perf_df = self.get_stats_df()

        del df, float_cols, filtered_df, grouped_df, merged_df, concat_df, new_df
        gc.collect()

        return perf_df


class PandasBench(PerformanceTracker):
    def __init__(self, args) -> None:
        logger.critical(f"{self.__class__.__name__}: Importing modules")
        self.pd = __import__("pandas")
        super().__init__(args)

    @profile
    def read_csv(self, path):
        df = self.pd.read_csv(path)
        return df

    @profile
    def add_column(self, df, array):
        df["rand_nums"] = array
        return df

    @profile
    def get_date_range(self):
        pd_dates = self.pd.date_range(start="1990-01-01", end="2050-12-31")
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
        return self.pd.merge(left, right, on=[on], how="left")

    @profile
    def groupby_merge(self, df, groupby_col, agg_col):
        grouped = df.groupby([groupby_col], as_index=False).agg(
            agg_mean=(f"{agg_col}", "mean"), agg_sum=(f"{agg_col}", "sum"), agg_std=(f"{agg_col}", "std")
        )
        return pd.merge(df, grouped, on=[groupby_col], how="left")

    @profile
    def concat(self, df_1, df_2):
        return self.pd.concat([df_1, df_2], axis=0)

    @profile
    def fill_na(self, df):
        df.fillna(0)

    @profile
    def drop_na(self, df):
        df.dropna()

    @profile
    def create_df(self, df_dict):
        return self.pd.DataFrame(df_dict)

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
        logger.critical(f"{self.__class__.__name__}: Importing modules")
        self.md = __import__("modin.pandas", fromlist=["pandas"])
        self.ray = __import__("ray")
        self.ray.init(runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}}, include_dashboard=False)

        PerformanceTracker.__init__(self, args)

    @profile
    def create_df(self, df_dict):
        return self.md.DataFrame(df_dict)

    @profile
    def read_csv(self, path):
        df = self.md.read_csv(path)
        return df

    @profile
    def get_date_range(self):
        pd_dates = self.md.date_range(start="1990-01-01", end="2050-12-31")
        return pd_dates

    @profile
    def merge(self, left, right, on):
        return self.md.merge(left, right, on=[on], how="left")

    @profile
    def groupby_merge(self, df, groupby_col, agg_col):
        grouped = df.groupby([groupby_col], as_index=False).agg(
            agg_mean=(f"{agg_col}", "mean"), agg_sum=(f"{agg_col}", "sum"), agg_std=(f"{agg_col}", "std")
        )
        return self.md.merge(df, grouped, on=[groupby_col], how="left")

    @profile
    def concat(self, df_1, df_2):
        return self.md.concat([df_1, df_2], axis=0)

    def get_stats_df(self):
        stats_df = super().get_stats_df()
        stats_df["framework"] = "modin"
        return stats_df
    
    def run_operations(self):
        perf_df = super().run_operations()
        if self.ray.is_initialized():
            logger.critical("Shutting down ray")
            self.ray.shutdown()
            
        return perf_df


class PolarsBench(PerformanceTracker):
    def __init__(self, args) -> None:
        logger.critical(f"{self.__class__.__name__}: Importing modules")
        self.pl = __import__("polars")
        super().__init__(args)

    @profile
    def read_csv(self, path):
        df = self.pl.read_csv(path)
        return df

    @profile
    def add_column(self, df, array):
        df = df.with_columns([self.pl.Series(array).alias("rand_num")])
        return df

    @profile
    def get_date_range(self):
        pl_dates = self.pl.date_range(
            low=datetime(1990, 1, 1), high=datetime(2050, 12, 31), interval="1d", closed="left"
        )
        return pl_dates

    @profile
    def filter_vals(self, df, filter_col, filter_val):
        return df.filter(self.pl.col(filter_col) > filter_val)

    @profile
    def groupby(self, df, groupby_col, agg_col):
        q = (
            df.lazy()
            .groupby(groupby_col)
            .agg(
                [
                    self.pl.col(agg_col).sum().suffix("_sum"),
                    self.pl.col(agg_col).mean().suffix("_mean"),
                    self.pl.col(agg_col).std().suffix("_std"),
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
                self.pl.all(),
                self.pl.col(agg_col).sum().over(groupby_col).alias(f"{agg_col}_sum"),
                self.pl.col(agg_col).mean().over(groupby_col).alias(f"{agg_col}_mean"),
            ]
        )

    @profile
    def concat(self, df_1, df_2):
        return self.pl.concat([df_1, df_2], how="diagonal")

    @profile
    def fill_na(self, df):
        df.fill_null(0)

    @profile
    def drop_na(self, df):
        df.drop_nulls()

    @profile
    def create_df(self, df_dict):
        return self.pl.from_dict(df_dict)

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
