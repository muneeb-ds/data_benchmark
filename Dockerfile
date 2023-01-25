FROM python:3.9-slim
RUN apt-get update
RUN apt-get -y install gcc python3-dev
RUN python -m pip install --upgrade pip
WORKDIR /data_benchmark
COPY *.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python", "benchmark.py" ]