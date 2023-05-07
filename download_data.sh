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

# download NATOPS
if [ ! -d "data/NATOPS" ]; then
    mkdir data/NATOPS/
    wget -P data/NATOPS http://timeseriesclassification.com/Downloads/NATOPS.zip
    unzip data/NATOPS/NATOPS.zip -d data/NATOPS/
    rm data/NATOPS/*.zip data/NATOPS/*.jpg
fi

# download AWR
if [ ! -d "data/ArticularyWordRecognition" ]; then
    mkdir data/ArticularyWordRecognition/
    wget -P data/ArticularyWordRecognition http://timeseriesclassification.com/Downloads/ArticularyWordRecognition.zip
    unzip data/ArticularyWordRecognition/ArticularyWordRecognition.zip -d data/ArticularyWordRecognition/
    rm data/ArticularyWordRecognition/*.zip data/ArticularyWordRecognition/*.jpg
fi

# download FaceDetection
if [ ! -d "data/FaceDetection" ]; then
    mkdir data/FaceDetection/
    wget -P data/FaceDetection http://timeseriesclassification.com/Downloads/FaceDetection.zip
    unzip data/FaceDetection/FaceDetection.zip -d data/FaceDetection/
    rm data/FaceDetection/*.zip data/FaceDetection/*.jpg
fi

# download PhonemeSpectra
if [ ! -d "data/PhonemeSpectra" ]; then
    mkdir data/PhonemeSpectra/
    wget -P data/PhonemeSpectra http://timeseriesclassification.com/Downloads/PhonemeSpectra.zip
    unzip data/PhonemeSpectra/PhonemeSpectra.zip -d data/PhonemeSpectra/
    rm data/PhonemeSpectra/*.zip data/PhonemeSpectra/*.jpg
fi

# download InsectWingbeat
if [ ! -d "data/InsectWingbeat" ]; then
    mkdir data/InsectWingbeat/
    wget -P data/InsectWingbeat http://timeseriesclassification.com/Downloads/InsectWingbeat.zip
    unzip data/InsectWingbeat/InsectWingbeat.zip -d data/InsectWingbeat/
    rm data/InsectWingbeat/*.zip data/InsectWingbeat/*.jpg data/InsectWingbeat/*.txt
fi

# download FingerMovements
if [ ! -d "data/FingerMovements" ]; then
    mkdir data/FingerMovements/
    wget -P data/FingerMovements http://timeseriesclassification.com/Downloads/FingerMovements.zip
    unzip data/FingerMovements/FingerMovements.zip -d data/FingerMovements/
    rm data/FingerMovements/*.zip data/FingerMovements/*.jpg data/FingerMovements/*.txt
fi
