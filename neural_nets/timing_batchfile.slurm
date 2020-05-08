#!/bin/bash
###ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE        #Do not propagate environment
#SBATCH --get-user-env=L     #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Nandy_yo     #Set the job name to "JobExample1"
#SBATCH --time=01:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=16000M                #Request 2560MB (2.5GB) per node
#SBATCH --output=time_linenet.txt    #Send stdout/err to "Example1Out.[jobID]"

#SBATCH --account=122802423399
#First Executable Line
cd $SCRATCH
module load Anaconda/3-5.0.0.1
source activate deeplearning
python /scratch/user/narendra5/LER_machine_learning/neural_nets/Timing_analysis_hprc.py
