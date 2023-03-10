from abc import ABC, abstractmethod
from datetime import datetime
import gc
import time
import logging
import numpy as np
import pandas as pd

from utils import profile, create_dataframe_dict, remove_parquets

logger = logging.getLogger(__name__)


class PerformanceTracker(ABC):
    def __init__(self, args) -> None:
        self.performance_dict = {}
        self.data_path = args.data_path
        self.row_size = args.rows
        self.column_size = args.columns
        self.iters = args.iterations
        self.performance_df = None
        self.pd = self.md = self.pl = self.ray = None
        self.conn = None

    def add(self, stats, operation):
        self.performance_dict[operation] = {}
        self.performance_dict[operation]["memory_usage (MB)"] = stats[0]
        self.performance_dict[operation]["time_consumed (S)"] = stats[1]
        return self

    def get_operation_stat(self, operation, func, *args):
        logger.critical("%s: %s", self.__class__.__name__, operation)
        output, stats = func(*args)
        self.add(stats, operation)
        return output

    def get_stats_df(self):
        self.performance_df = pd.DataFrame.from_dict(self.performance_dict, orient="index")
        self.performance_df["framework"] = self.__class__.__name__.split("Bench", maxsplit=1)[
            0
        ].lower()
        return self.performance_df

    @abstractmethod
    def read_csv(self, path):
        pass

    @profile
    def get_shape(self, df):
        return df.shape

    @abstractmethod
    def read_parquet(self, path):
        pass

    @abstractmethod
    def add_column(self, df, array):
        pass

    @abstractmethod
    def get_date_range(self):
        pass

    @profile
    def get_float_cols(self, df):
        float_cols = [
            col
            for col in df.columns
            if str(df[col].dtype) in ["float", "Float64", "Float16", "float64"]
        ]

        return float_cols
    
    @profile
    def get_str_cols(self, df):
        return [col for col in df.columns if str(df[col].dtype) in ["object", "str", "Utf8"]]

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

    def log_total_time(self, start, end):
        t_final = end - start
        operation = "log total time"
        logger.critical("%s: %s", self.__class__.__name__, operation)
        self.add((np.NaN, t_final), operation)

    def run_operations(self):
        operation = "reading csv"
        df = self.get_operation_stat(operation, self.read_csv, self.data_path)

        operation = "get shape"
        shape = self.get_operation_stat(operation, self.get_shape, df)

        rand_arr = np.random.randint(0, 100, shape[0])

        operation = "add column"
        _ = self.get_operation_stat(operation, self.add_column, df, rand_arr)

        operation = "get date range"
        _ = self.get_operation_stat(operation, self.get_date_range)

        operation = "get float columns"
        float_cols = self.get_operation_stat(operation, self.get_float_cols, df)

        filter_col = np.random.choice(float_cols)

        operation = "get column mean val"
        filter_val = self.get_operation_stat(operation, self.col_mean, df, filter_col)

        operation = "filter values based on col mean"
        filtered_df = self.get_operation_stat(
            operation, self.filter_vals, df, filter_col, filter_val
        )

        operation = "get string columns"
        df_str_cols = self.get_operation_stat(operation, self.get_str_cols, df)
        groupby_col = np.random.choice(df_str_cols)

        operation = "groupby aggregation (sum, mean, std)"
        grouped_df = self.get_operation_stat(operation, self.groupby, df, groupby_col, filter_col)

        operation = "merging grouped col to original df"
        merged_df = self.get_operation_stat(operation, self.merge, df, grouped_df, groupby_col)

        operation = "combined groupby merge"
        merged_df = self.get_operation_stat(
            operation, self.groupby_merge, df, groupby_col, filter_col
        )

        operation = "horizontal concatenatenation"
        concat_df = self.get_operation_stat(operation, self.concat, merged_df, filtered_df)

        operation = "fill nulls with 0"
        concat_df = self.get_operation_stat(operation, self.fill_na, concat_df)

        operation = "drop nulls"
        concat_df = self.get_operation_stat(operation, self.drop_na, concat_df)

        operation = "describe stats of df"
        _ = self.get_operation_stat(operation, self.describe_df, concat_df)

        parquet_path = "sample_data.parquet"
        remove_parquets(parquet_path)

        try:
            concat_df[concat_df.select_dtypes(include=[object]).columns] = concat_df[
                concat_df.select_dtypes(include=[object]).columns
            ].astype(str)
        except AttributeError:
            pass

        operation = "save to csv"
        _ = self.get_operation_stat(operation, self.save_to_csv, concat_df)

        operation = "save to parquet"
        _ = self.get_operation_stat(operation, self.save_to_parquet, concat_df)

        operation = "read from parquet"
        _ = self.get_operation_stat(operation, self.read_parquet, parquet_path)

        df_dict = create_dataframe_dict(self.row_size, self.column_size)

        operation = f"create dataframe of size: ({self.row_size},{self.column_size})"
        new_df = self.get_operation_stat(operation, self.create_df, df_dict)

        t_final = time.perf_counter()

        del (
            df,
            float_cols,
            filtered_df,
            grouped_df,
            merged_df,
            concat_df,
            new_df,
            rand_arr,
        )
        gc.collect()

        return t_final

    def run(self):
        time_0 = time.perf_counter()
        t_final = self.run_operations()

        self.log_total_time(time_0, t_final)

        logger.critical("%s: combining stats", self.__class__.__name__)

        perf_df = self.get_stats_df()

        return perf_df


class PandasBench(PerformanceTracker):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.pd = None

    @profile
    def read_csv(self, path):
        df = self.pd.read_csv(path)
        return df

    @profile
    def read_parquet(self, path):
        df = self.pd.read_parquet(path)
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
            agg_mean=(f"{agg_col}", "mean"),
            agg_sum=(f"{agg_col}", "sum"),
            agg_std=(f"{agg_col}", "std"),
        )

    @profile
    def merge(self, left, right, on):
        return self.pd.merge(left, right, on=[on], how="left")

    @profile
    def groupby_merge(self, df, groupby_col, agg_col):
        grouped = df.groupby([groupby_col], as_index=False).agg(
            agg_mean=(f"{agg_col}", "mean"),
            agg_sum=(f"{agg_col}", "sum"),
            agg_std=(f"{agg_col}", "std"),
        )
        return pd.merge(df, grouped, on=[groupby_col], how="left")

    @profile
    def concat(self, df_1, df_2):
        df_concat = self.pd.concat([df_1, df_2], axis=0)
        return df_concat

    @profile
    def fill_na(self, df):
        return df.fillna(0)

    @profile
    def drop_na(self, df):
        return df.dropna()

    @profile
    def create_df(self, df_dict):
        return self.pd.DataFrame(df_dict)

    @profile
    def save_to_csv(self, df):
        df.to_csv("sample_data.csv", index=False)

    @profile
    def save_to_parquet(self, df):
        df.to_parquet("sample_data.parquet", index=False, engine="pyarrow")

    def run_operations(self):
        logger.critical("%s: Importing modules", self.__class__.__name__)
        self.pd = __import__("pandas")
        return super().run_operations()


class ModinBench(PerformanceTracker):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.md = self.ray = None

    @profile
    def read_csv(self, path):
        df = self.md.read_csv(path)
        return df

    @profile
    def read_parquet(self, path):
        df = self.md.read_parquet(path)
        return df

    @profile
    def add_column(self, df, array):
        df["rand_nums"] = array
        return df

    @profile
    def get_date_range(self):
        pd_dates = self.md.date_range(start="1990-01-01", end="2050-12-31")
        return pd_dates

    @profile
    def filter_vals(self, df, filter_col, filter_val):
        return df.loc[df[filter_col] > filter_val, :]

    @profile
    def groupby(self, df, groupby_col, agg_col):
        return df.groupby([groupby_col], as_index=False).agg(
            agg_mean=(f"{agg_col}", "mean"),
            agg_sum=(f"{agg_col}", "sum"),
            agg_std=(f"{agg_col}", "std"),
        )

    @profile
    def merge(self, left, right, on):
        return self.md.merge(left, right, on=[on], how="left")

    @profile
    def groupby_merge(self, df, groupby_col, agg_col):
        grouped = df.groupby([groupby_col], as_index=False).agg(
            agg_mean=(f"{agg_col}", "mean"),
            agg_sum=(f"{agg_col}", "sum"),
            agg_std=(f"{agg_col}", "std"),
        )
        return self.md.merge(df, grouped, on=[groupby_col], how="left")

    @profile
    def concat(self, df_1, df_2):
        return self.md.concat([df_1, df_2], axis=0)

    @profile
    def fill_na(self, df):
        return df.fillna(0)

    @profile
    def drop_na(self, df):
        return df.dropna()

    @profile
    def create_df(self, df_dict):
        return self.md.DataFrame(df_dict)

    @profile
    def save_to_csv(self, df):
        df.to_csv("sample_data.csv", index=False)

    @profile
    def save_to_parquet(self, df):
        df.to_parquet("sample_data.parquet", index=False, engine="pyarrow")

    def run_operations(self):
        logger.critical("%s: Importing modules", self.__class__.__name__)
        self.md = __import__("modin.pandas", fromlist=["pandas"])
        self.ray = __import__("ray")
        self.ray.init(
            runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}},
            include_dashboard=False,
            ignore_reinit_error=True,
        )
        perf_df = super().run_operations()
        return perf_df


class PolarsBench(PerformanceTracker):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.pl = None

    @profile
    def read_csv(self, path):
        df = self.pl.read_csv(path)
        return df

    @profile
    def read_parquet(self, path):
        df = self.pl.read_parquet(path)
        return df

    @profile
    def add_column(self, df, array):
        df = df.with_columns([self.pl.Series(array).alias("rand_num")])
        return df

    @profile
    def get_date_range(self):
        pl_dates = self.pl.date_range(
            low=datetime(1990, 1, 1),
            high=datetime(2050, 12, 31),
            interval="1d",
            closed="left",
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
        return df.fill_null(0)

    @profile
    def drop_na(self, df):
        return df.drop_nulls()

    @profile
    def create_df(self, df_dict):
        return self.pl.from_dict(df_dict)

    @profile
    def save_to_csv(self, df):
        df.write_csv("sample_data.csv")

    @profile
    def save_to_parquet(self, df):
        df.write_parquet("sample_data.parquet")

    def run_operations(self):
        logger.critical("%s: Importing modules", self.__class__.__name__)
        self.pl = __import__("polars")
        return super().run_operations()


class DuckdbBench(PerformanceTracker):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.duckdb = self.conn = None
        self.pq = None

    @profile
    def read_csv(self, path):
        query = f"CREATE OR REPLACE TABLE dataframe AS SELECT * FROM read_csv_auto ('{path}')"
        self.conn.execute(query)
        return None
    
    @profile
    def get_shape(self, df):
        rows = self.conn.execute("SELECT COUNT(*) AS row_count FROM dataframe").fetchnumpy()['row_count'][0]
        columns = len(self.conn.execute("DESCRIBE TABLE dataframe").fetchnumpy()['column_name'])
        shape = (rows, columns)
        return shape

    @profile
    def add_column(self, df, array):
        self.conn.execute("ALTER TABLE dataframe ADD COLUMN rand_nums INTEGER")
        self.conn.execute("update dataframe set rand_nums = floor(random() * 100 + 1)::int")
        return None

    @profile
    def get_date_range(self):
        query = "SELECT t.day::date FROM generate_series(timestamp '1990-01-01', timestamp '2050-12-31', interval  '1 day') AS t(day);"
        dates = self.conn.execute(query).fetchnumpy()
        return dates

    @profile
    def get_float_cols(self, df):
        described_df = self.conn.execute("DESCRIBE TABLE dataframe").df()
        float_cols = described_df[described_df['column_type']=="DOUBLE"]['column_name'].unique()
        return float_cols
    
    @profile
    def get_str_cols(self, df):
        described_df = self.conn.execute("DESCRIBE TABLE dataframe").df()
        float_cols = described_df[described_df['column_type']=="VARCHAR"]['column_name'].unique()
        return float_cols

    @profile
    def col_mean(self, df, filter_col):
        filter_val = self.conn.execute(f"SELECT AVG({filter_col}) FROM dataframe").fetchall()[0][0]
        return filter_val

    @profile
    def filter_vals(self, df, filter_col, filter_val):
        self.conn.execute(
            f"CREATE OR REPLACE VIEW filtered_df AS SELECT * FROM dataframe WHERE {filter_col} > {filter_val}"
        )
        return None

    @profile
    def groupby(self, df, groupby_col, agg_col):
        self.conn.execute(
            f"CREATE OR REPLACE VIEW grouped_df AS SELECT {groupby_col}, SUM({agg_col}) AS {agg_col}_sum, AVG({agg_col}) AS {agg_col}_avg, STDDEV({agg_col}) AS {agg_col}_std FROM dataframe GROUP BY {groupby_col}"
        )
        return

    @profile
    def merge(self, left, right, on):
        self.conn.execute(
            f"CREATE OR REPLACE VIEW merged_df AS SELECT * FROM dataframe LEFT JOIN grouped_df ON dataframe.{on} = grouped_df.{on}"
        )
        return

    @profile
    def groupby_merge(self, df, groupby_col, agg_col):
        og_cols = self.conn.execute("DESCRIBE TABLE dataframe").fetchnumpy()['column_name']
        og_cols = [f"t1.{col}" for col in og_cols]
        og_cols = ", ".join(og_cols)
        self.conn.execute(
            f"CREATE OR REPLACE VIEW merged_df AS SELECT {og_cols}, {agg_col}_sum, {agg_col}_avg, {agg_col}_std FROM dataframe AS t1 LEFT JOIN (SELECT {groupby_col}, SUM({agg_col}) AS {agg_col}_sum, AVG({agg_col}) AS {agg_col}_avg, STDDEV({agg_col}) AS {agg_col}_std FROM dataframe GROUP BY {groupby_col}) AS t2 ON t1.{groupby_col}=t2.{groupby_col}"
        )
        return

    @profile
    def concat(self, df_1, df_2):
        self.conn.execute("CREATE OR REPLACE VIEW concat_table AS SELECT * FROM merged_df UNION ALL SELECT * FROM merged_df")
        return

    @profile
    def fill_na(self, df):
        df_cols = self.conn.execute("DESCRIBE TABLE concat_table").fetchnumpy()['column_name']
        query = "CREATE OR REPLACE VIEW concat_table_filled_nulls AS SELECT"
        for col in df_cols:
            query += f" COALESCE({col}, 0) AS {col},"
        query += " FROM concat_table"
        df = self.conn.execute(query)
        return

    @profile
    def drop_na(self, df):
        df_cols = self.conn.execute("DESCRIBE TABLE concat_table").fetchnumpy()['column_name']
        query = "CREATE OR REPLACE VIEW concat_table_dropped_nulls AS SELECT * FROM concat_table WHERE"
        for i, col in enumerate(df_cols):
            query += f" {col} IS NOT NULL"
            if i+1 != len(df_cols):
                query += f" AND"
        self.conn.execute(query)
        return

    @profile
    def describe_df(self, df):
        described = self.conn.execute("SUMMARIZE concat_table").df()
        return described

    @profile
    def save_to_csv(self, df):
        self.conn.execute("COPY concat_table TO 'sample_data_duck.csv' (HEADER, DELIMETER ',')")

    @profile
    def save_to_parquet(self, df):
        self.conn.execute("COPY concat_table TO 'sample_data.parquet' (FORMAT PARQUET)")

    @profile
    def read_parquet(self, path):
        self.conn.execute(
            f"CREATE OR REPLACE VIEW parquet_df AS SELECT * FROM read_parquet({path})"
        )

    @profile
    def create_df(self, df_dict):
        df_from_dict = pd.DataFrame.from_dict(df_dict)
        self.conn.execute("CREATE OR REPLACE VIEW created_df AS SELECT * FROM df_from_dict")

    @profile
    def convert_to_pandas(self):
        df = self.conn.execute("SELECT * FROM dataframe").df()
        return df

    @profile
    def convert_to_numpy(self):
        df = self.conn.execute("SELECT * FROM dataframe").fetchnumpy()
        return df

    @profile
    def convert_to_arrow(self):
        df = self.conn.execute("SELECT * FROM dataframe").arrow()
        return df

    def run_operations(self):
        logger.critical("%s: Importing modules", self.__class__.__name__)
        self.duckdb = __import__("duckdb")
        self.pl = __import__("polars")
        self.pq = __import__("pyarrow.parquet", fromlist="parquet")
        self.conn = self.duckdb.connect(":memory:")

        t_mid = super().run_operations()

        t_start = time.perf_counter()
        operation = "converting to pandas"
        pd_df = self.get_operation_stat(operation, self.convert_to_pandas)
        del pd_df

        operation = "converting to numpy"
        np_df = self.get_operation_stat(operation, self.convert_to_numpy)
        del np_df

        operation = "converting to arrow"
        arrow_df = self.get_operation_stat(operation, self.convert_to_arrow)
        del arrow_df

        t_end = time.perf_counter() - t_start
        
        t_final = t_end + t_mid
        gc.collect()
        return t_final
