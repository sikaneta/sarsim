#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate radar
python -m generateMsar --config-xml /home/ishuwa/simulation/40cm/simulation_40cm.xml --channel-idxs $1 --target-rangeidx 400 --rblock-size 400
