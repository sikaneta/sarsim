#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 0 256 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 256 512 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 512 768 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 768 1024 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 1024 1280 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 1280 1536 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 1536 1792 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 1792 2048 --xblock-size 16384 &
wait
