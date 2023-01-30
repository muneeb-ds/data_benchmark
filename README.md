# Benchmarking for data wrangling frameworks
Currently supported APIs: **Pandas, Modin, Polars**

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
