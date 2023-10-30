#!/bin/bash

PY=python
DATASET="$1"

$PY src/main.py --dataset $DATASET \
                --label "EXP7" \
                --startdate "2020-07-01" \
                --enddate "2021-01-01" \
                --mink 2 \
                --maxk 10 \
                --features 10 \
                --minaccans 3 \
                --ltrsize 50000 \
                --probability 0.001 \
                --restart 5 \
                --steps 10 \
                --maxevals 300 \
                --testsize 5000 
