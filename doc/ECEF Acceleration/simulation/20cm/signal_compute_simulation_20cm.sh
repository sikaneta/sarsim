#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 0 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 1 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 2 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 3 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 4 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 5 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 6 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 7 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 8 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 9 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 10 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 11 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 12 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 13 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 14 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 15 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 16 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 17 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 18 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 19 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 20 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 21 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 22 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 23 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 24 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 25 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 26 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 27 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 28 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 29 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 30 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 31 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 32 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 33 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 34 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 35 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 36 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 37 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 38 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 39 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 40 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 41 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 42 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 43 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 44 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 45 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 46 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 47 --target-rangeidx 400 --rblock-size 256
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs 48 --target-rangeidx 400 --rblock-size 256
