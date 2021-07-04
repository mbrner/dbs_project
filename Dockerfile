FROM continuumio/miniconda3
RUN apt update && apt install -y build-essential

RUN mkdir -p /dbs_project

COPY docker_config.toml /dbs_project/docker_config.toml
COPY requirements.txt /dbs_project/requirements.txt
COPY data /dbs_project/data

RUN pip install -r /dbs_project/requirements.txt

COPY src /dbs_project/src
WORKDIR /dbs_project/src
CMD python /dbs_project/src/start.py /dbs_project/docker_config.toml