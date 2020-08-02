#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 524288 540672 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 540672 557056 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 557056 573440 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 573440 589824 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 589824 606208 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 606208 622592 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 622592 638976 --rblock-size 256 &
python -m sarProcess --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --xidx 638976 655360 --rblock-size 256 &
wait
