#!/bin/bash
pkill -9 tritonserver
echo > tritonserver.log
nohup start_triton.sh >> tritonserver.log &
