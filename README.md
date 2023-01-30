# Benchmarking for data wrangling frameworks
Currently supported APIs: **[Pandas](https://pandasguide.readthedocs.io/en/latest/), [Modin](https://modin.readthedocs.io/en/stable/), [Polars](https://pola-rs.github.io/polars-book/user-guide/)**

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

### Tracked statistics of an operation:
- Peak memory utilization 
- Time consumption

# Usage
***Make sure data being read has atleast 1 string column and 1 float column***

## Run with Docker (Recommended)
```
1. docker build -t [image-name/tag] .
2. docker run -v [local/data/storage/dir]:[/container/data/dir] [image-name] \
   --data_path [/container/data/dir/data.csv] \
   --save_dir [/container/data/dir] \
   --iterations [number of times to run benchmarking] \
   --frameworks [supported framework names i.e. pandas modin polars]
```
***if you want to access the output of the tests, make sure that the save_dir is the same as the mounted dir on container***

## Run on venv/conda
```
1. pip install -r requirements.txt
2. python benchmark.py --data_path [data/dir/file.csv]
```

Check [here](https://github.com/muneeb-ds/data_benchmark/blob/460692d675a4da092d0ac722c2e3aa59119df44b/utils.py#L12-L23) for all arguments

## To add new operations:
1. Go to [operations.py](https://github.com/muneeb-ds/data_benchmark/blob/main/operations.py)
2. Underneath the [PerformanceBenchmark](https://github.com/muneeb-ds/data_benchmark/blob/a22f3af8e75f45d13c626856e943e56ce443d673/operations.py#L14) base class, add an abstract method for your operation
3. Add the same method to all subclasses and their corresponsing functionality underneath
4. Pass the operation name, method and any args to [get_operation_stat](https://github.com/muneeb-ds/data_benchmark/blob/a22f3af8e75f45d13c626856e943e56ce443d673/operations.py#L28) method and pass this method underneath the [run_operations](https://github.com/muneeb-ds/data_benchmark/blob/a22f3af8e75f45d13c626856e943e56ce443d673/operations.py#L104) method in base class
