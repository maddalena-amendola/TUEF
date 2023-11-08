#!/bin/bash

PY=python
DATASET="$1"
LABEL="EXP7"
STARTDATE="2020-07-01"
ENDDATE="2021-01-01"
MINK=2
MAXK=10
FEATURES=10
MINACCANS=3
LTRSIZE=50000
PROB=0.001
RESTART=5
STEPS=10
MAXEVALS=300
TESTSIZE=5000

$PY Baselines/TUEFCB/TUEFCB.py --dataset $DATASET \
                --label $LABEL \
                --ltrsize $LTRSIZE \
                --maxevals $MAXEVALS \
                --testsize $TESTSIZE 

$PY Baselines/TUEFNB/TUEFNB.py --dataset $DATASET \
                --label $LABEL \
                --ltrsize $LTRSIZE \
                --maxevals $MAXEVALS \
                --testsize $TESTSIZE 

$PY Baselines/TUEFNORW/TUEFNORW.py --dataset $DATASET \
                --label $LABEL \
                --ltrsize $LTRSIZE \
                --maxevals $MAXEVALS \
                --testsize $TESTSIZE 

$PY Baselines/TUEFSL/TUEFSL.py --dataset $DATASET \
                --label $LABEL \
                --startdate $STARTDATE \
                --enddate $ENDDATE \
                --features $FEATURES \
                --minaccans $MINACCANS \
                --ltrsize $LTRSIZE \
                --probability $PROB \
                --restart $RESTART \
                --steps $STEPS \
                --maxevals $MAXEVALS \
                --testsize $TESTSIZE 

$PY Baselines/BC/BC.py --dataset $DATASET \
                --label $LABEL \
                --testsize $TESTSIZE 

$PY Baselines/BM25/BM25.py --dataset $DATASET \
                --label $LABEL \
                --testsize $TESTSIZE 

$PY Baselines/TUEFLIN/TUEFLIN.py --dataset $DATASET \
                --label $LABEL \
                --testsize $TESTSIZE 
