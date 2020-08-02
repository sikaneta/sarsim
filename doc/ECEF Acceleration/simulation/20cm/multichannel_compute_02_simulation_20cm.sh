#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 4096 4352 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 4352 4608 --xblock-size 16384 &
wait
