#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 0 16384 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 16384 32768 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 32768 49152 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 49152 65536 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 65536 81920 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 81920 98304 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 98304 114688 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 114688 131072 --rblock-size 400 &
wait
