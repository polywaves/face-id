#!/bin/bash
python3 -m grpc_tools.protoc -I ./proto-specs --python_out=. --grpc_python_out=. ./proto-specs/cameras.proto
python3 recognize.py