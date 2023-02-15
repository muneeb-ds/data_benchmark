# Benchmarking for data wrangling frameworks
Currently supported APIs: **[Pandas](https://pandasguide.readthedocs.io/en/latest/), [Modin](https://modin.readthedocs.io/en/stable/), [Polars](https://pola-rs.github.io/polars-book/user-guide/), [DuckDB](https://duckdb.org/docs/api/python/overview), [SparkPandas](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)**

### Operations Tested:
- read csv
- read parquet
- add column of np arrays
- get date range
- filter rows based on random value
- groupby
- merge
- concatenation
- fill/drop nulls
- create dataframe
- save to csv
- save to parquet
- convert to pandas/numpy/arrow (duckdb only)

### Tracked statistics of an operation:
- Peak memory utilization 
- Time consumption

# Usage
***Make sure data being read has atleast 1 string column and 1 float column***


## Run on venv/conda
```
1. pip install -r requirements.txt
2. python benchmark.py --data_path [data/dir/file.csv]
```

**Note: if you want to use pandas-on-spark, make sure you have [java](https://www.java.com/en/download/) installed**

Check [here](https://github.com/muneeb-ds/data_benchmark/blob/460692d675a4da092d0ac722c2e3aa59119df44b/utils.py#L12-L23) for all arguments

## Run with Docker
```
1. docker build -t [image-name/tag] .
2. docker run -v [local/data/storage/dir]:[/container/data/dir] [image-name] \
   --data_path [/container/data/dir/data.csv] \
   --save_dir [/container/data/dir] \
   --iterations [number of times to run benchmarking] \
   --frameworks [supported framework names i.e. pandas modin polars]
```
***if you want to access the output of the tests, make sure that the save_dir is the same as the mounted dir on container***

# Further Testing

## To add new operations for all frameworks:
1. Go to [operations.py](https://github.com/muneeb-ds/data_benchmark/blob/main/operations.py)
2. Underneath the [PerformanceBenchmark](https://github.com/muneeb-ds/data_benchmark/blob/a22f3af8e75f45d13c626856e943e56ce443d673/operations.py#L14) base class, add an abstract method for your operation
3. Add the same method to all subclasses and their corresponsing functionality underneath
4. Pass the operation name, method and any args to [get_operation_stat](https://github.com/muneeb-ds/data_benchmark/blob/a22f3af8e75f45d13c626856e943e56ce443d673/operations.py#L28) method and pass this method underneath the [run_operations](https://github.com/muneeb-ds/data_benchmark/blob/a22f3af8e75f45d13c626856e943e56ce443d673/operations.py#L104) method in base class

## To only add operations for a specific framework:
1. Define an operation within the frameworks class with the @profile decorator
2. Underneath the framework's class define a run_operations method
3. Structure the method to call your operations plus all common operations with (super().run_operations())
4. Return the total time taken to run operations

## To add new frameworks to test:
1. Go to [operations.py](https://github.com/muneeb-ds/data_benchmark/blob/main/operations.py)
2. Create a class with the name of your framework followed by **Bench** for example: `class FrameworkBench:`
3. Inherit this class from [PerformanceBenchmark](https://github.com/muneeb-ds/data_benchmark/blob/a22f3af8e75f45d13c626856e943e56ce443d673/operations.py#L14)
4. Add all the methods used in [run_operations](https://github.com/muneeb-ds/data_benchmark/blob/a22f3af8e75f45d13c626856e943e56ce443d673/operations.py#L104) method
5. Include framework specific functionality for each method
6. Add framework name in lowercase as one of the choices [here](https://github.com/muneeb-ds/data_benchmark/blob/fd868d69bdf98591f4bc9e3ebc53504b1a0069f9/utils.py#L22)
