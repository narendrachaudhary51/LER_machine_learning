#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE        #Do not propagate environment
#SBATCH --get-user-env=L     #Replicate login environment
  
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=linenet_nodrop     #Set the job name to "JobExample1"
#SBATCH --time=47:55:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=64000M                #Request 2560MB (2.5GB) per node
#SBATCH --output=Linenet_round_L2_nodrop_epoch4.%j    #Send stdout/err to "Example1Out.[jobID]"

#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --account=122802429018
#First Executable Line
cd $SCRATCH
module load Anaconda/3-5.0.0.1
source activate deeplearning
python /scratch/user/narendra5/LER_machine_learning/neural_nets/LINEnet.py
