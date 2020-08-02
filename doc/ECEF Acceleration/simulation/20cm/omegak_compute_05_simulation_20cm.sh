#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 655360 671744 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 671744 688128 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 688128 704512 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 704512 720896 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 720896 737280 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 737280 753664 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 753664 770048 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 770048 786432 --rblock-size 256 &
wait
