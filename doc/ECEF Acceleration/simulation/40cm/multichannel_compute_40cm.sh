#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
wait
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/40cm.xml --ridx 0 400 --xblock-size 8192 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/40cm.xml --ridx 400 800 --xblock-size 8192 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/40cm.xml --ridx 800 1024 --xblock-size 8192 &
wait
