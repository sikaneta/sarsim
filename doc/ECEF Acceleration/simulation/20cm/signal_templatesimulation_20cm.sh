#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate radar
python -m generateMsar --config-xml /home/ishuwa/simulation/20cm/simulation_20cm.xml --channel-idxs $1 --target-rangeidx 400 --rblock-size 256
