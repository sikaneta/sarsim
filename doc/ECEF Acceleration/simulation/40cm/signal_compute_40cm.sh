#!/bin/bash
#SBATCH -J sarsim
#SBATCH -N 1
eval "$(conda shell.bash hook)"
conda activate radar
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 0 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 1 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 2 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 3 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 4 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 5 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 6 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 7 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 8 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 9 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 10 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 11 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 12 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 13 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 14 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 15 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 16 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 17 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 18 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 19 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 20 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 21 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 22 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 23 --target-rangeidx 400 --rblock-size 400
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/40cm.xml --channel-idxs 24 --target-rangeidx 400 --rblock-size 400
