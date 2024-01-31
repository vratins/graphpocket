#!/usr/bin/env bash

if [ -z "$STRUCTURE_DIR" ]; then
  echo "STRUCTURE_DIR not set"
  exit
fi
if ! type wget > /dev/null; then
  echo "wget not installed"
  exit
fi
if ! type unzip > /dev/null; then
  echo "unzip not installed"
  exit
fi

cd $STRUCTURE_DATA_DIR

# TOUGH-M1 dataset
mkdir TOUGH-M1
wget https://zenodo.org/record/3687317/files/dt_tough.zip?download=1 -O dt_tough.zip && unzip dt_tough.zip
rm dt_tough.zip
wget https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/L7H7JJ/UFO5CB -O official_tough_m1.tar.gz && tar -xvzf official_tough_m1.tar.gz -C TOUGH-M1
rm official_tough_m1.tar.gz
wget https://osf.io/tmgne/download -O TOUGH-M1/TOUGH-M1_positive.list
wget https://osf.io/6dn5s/download -O TOUGH-M1/TOUGH-M1_pocket.list
wget https://osf.io/3aypv/download -O TOUGH-M1/TOUGH-M1_negative.list



