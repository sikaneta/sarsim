#!/bin/bash
#SBATCH -J sar_array
#SBATCH -N 1
#SBATCH --array 0-24:1
/home/ishuwa/simulation/40cm/signal_templatesimulation_40cm.sh $SLURM_ARRAY_TASK_ID
