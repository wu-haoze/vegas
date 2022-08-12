#!/usr/bin/env bash

# Decima requires python 3.6
if [! -d py3 ]
do
    virtualenv -p python3.6 py3
done

if [ -d py3 ]
do
    echo Successfully created virtual environment
done

source py3/bin/activate

#pip3 install requirements.txt

git clone https://github.com/anwu1219/Marabou --depth=1
cd Marabou
mkdir build
cd build
cmake ../ -DENABLE_GUROBI=ON
make -j12
