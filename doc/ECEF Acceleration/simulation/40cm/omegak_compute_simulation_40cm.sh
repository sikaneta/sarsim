#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 0 16384 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 16384 32768 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 32768 49152 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 49152 65536 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 65536 81920 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 81920 98304 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 98304 114688 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 114688 131072 --rblock-size 400 &
wait
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 131072 147456 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 147456 163840 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 163840 180224 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 180224 196608 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 196608 212992 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 212992 229376 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 229376 245760 --rblock-size 400 &
python -m sarProcess --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --xidx 245760 246471 --rblock-size 400 &
wait
