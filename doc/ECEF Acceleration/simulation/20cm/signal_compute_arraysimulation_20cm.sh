#!/bin/bash
#SBATCH -J sar_array
#SBATCH -N 1
#SBATCH --array 0-48:1
/home/ishuwa/simulation/20cm/signal_templatesimulation_20cm.sh $SLURM_ARRAY_TASK_ID
