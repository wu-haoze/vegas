#!/usr/bin/env bash

# Decima requires python 3.6
if [ ! -d py3 ]
then
    virtualenv -p python3.7 py3
fi

if [ -d py3 ]
then
    echo Successfully created virtual environment
fi

source py3/bin/activate

# reinstall pip
curl -sS https://bootstrap.pypa.io/get-pip.py  -o get-pip.py
python get-pip.py --force-reinstall

pip3 install -r requirements.txt

git clone https://github.com/anwu1219/Marabou
cd Marabou
git fetch origin vmware
git checkout vmware
mkdir build
cd build
cmake ../ -DENABLE_GUROBI=ON
make -j12
