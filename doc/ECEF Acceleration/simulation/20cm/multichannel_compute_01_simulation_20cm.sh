#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 2048 2304 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 2304 2560 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 2560 2816 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 2816 3072 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 3072 3328 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 3328 3584 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 3584 3840 --xblock-size 16384 &
python -m processMultiChannel --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --ridx 3840 4096 --xblock-size 16384 &
wait
