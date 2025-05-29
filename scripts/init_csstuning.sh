#!/bin/bash
pip install prismbo_external/csstuning

bash prismbo_external/csstuning/cssbench/compiler/docker/build_docker.sh
bash prismbo_external/csstuning/cssbench/dbms/docker/build_docker.sh

csstuning_dbms_init -h
