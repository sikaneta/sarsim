#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 131072 147456 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 147456 163840 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 163840 180224 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 180224 196608 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 196608 212992 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 212992 229376 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 229376 245760 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 245760 246471 --rblock-size 400 &
wait
