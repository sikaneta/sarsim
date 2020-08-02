#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --ridx 0 400 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --ridx 400 800 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --ridx 800 1024 --xblock-size 16384 &
wait
