#!/bin/bash

if [ ! -d data/ ]; then
    mkdir data/
fi

# download LSST
if [ ! -d "data/LSST/" ]; then
    mkdir data/LSST/
    wget -P data/LSST/ http://timeseriesclassification.com/Downloads/LSST.zip
    unzip data/LSST/LSST.zip -d data/LSST/
    rm data/LSST/*.txt data/LSST/*.JPG data/LSST/*.ts
fi


# download Tiselac
if [ ! -d "data/Tiselac" ]; then
    mkdir data/Tiselac/
    wget -P data/Tiselac http://timeseriesclassification.com/Downloads/Tiselac.zip
    unzip data/Tiselac/Tiselac.zip -d data/Tiselac/
    rm data/Tiselac/*.zip data/Tiselac/*.jpg
fi

# download SpokenArabicDigits
if [ ! -d "data/SpokenArabicDigits" ]; then
    mkdir data/SpokenArabicDigits/
    wget -P data/SpokenArabicDigits http://timeseriesclassification.com/Downloads/SpokenArabicDigits.zip
    unzip data/SpokenArabicDigits/SpokenArabicDigits.zip -d data/SpokenArabicDigits/
    rm data/SpokenArabicDigits/*.zip data/SpokenArabicDigits/*.jpg
fi

# download FaceDetection
if [ ! -d "data/FaceDetection" ]; then
    mkdir data/FaceDetection/
    wget -P data/FaceDetection http://timeseriesclassification.com/Downloads/FaceDetection.zip
    unzip data/FaceDetection/FaceDetection.zip -d data/FaceDetection/
    rm data/FaceDetection/*.zip data/FaceDetection/*.jpg
fi
