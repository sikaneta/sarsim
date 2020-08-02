#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 262144 278528 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 278528 294912 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 294912 311296 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 311296 327680 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 327680 344064 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 344064 360448 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 360448 376832 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 376832 393216 --rblock-size 256 &
wait
