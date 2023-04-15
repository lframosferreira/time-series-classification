#!/bin/bash

mkdir data

# download LSST
mkdir data/LSST/
wget -P data/LSST/ http://timeseriesclassification.com/Downloads/LSST.zip
unzip data/LSST/LSST.zip -d data/LSST/
rm data/LSST/*.txt data/LSST/*.JPG data/LSST/*.ts