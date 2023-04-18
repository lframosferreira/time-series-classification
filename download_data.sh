#!/bin/bash

mkdir data

# download LSST
mkdir data/LSST/
wget -P data/LSST/ http://timeseriesclassification.com/Downloads/LSST.zip
unzip data/LSST/LSST.zip -d data/LSST/
rm data/LSST/*.txt data/LSST/*.JPG data/LSST/*.ts


# download Tiselac
mkdir data/Tiselac/
wget -P data/Tiselac http://timeseriesclassification.com/Downloads/Tiselac.zip
unzip data/Tiselac/Tiselac.zip -d data/Tiselac/
rm data/Tiselac/*.zip data/Tiselac/*.png