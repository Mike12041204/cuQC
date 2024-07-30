#!/bin/bash

./run.sh 1 fb-pages .9 24 dss_fb-pages.csv
./run.sh 2 fb-pages .9 24 dss_fb-pages.csv
./run.sh 10 fb-pages .9 24 dss_fb-pages.csv
./run.sh 50 fb-pages .9 24 dss_fb-pages.csv
./run.sh 100 fb-pages .9 24 dss_fb-pages.csv

./run.sh 1 konect .5 15 dss_konect.csv
./run.sh 2 konect .5 15 dss_konect.csv
./run.sh 10 konect .5 15 dss_konect.csv
./run.sh 50 konect .5 15 dss_konect.csv
./run.sh 100 konect .5 15 dss_konect.csv

