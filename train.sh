#! /bin/bash -l

# Job name
#$ -N DeNA_80

# Time limit
#$ -l h_rt=48:00:00

# Request 4 CPUs
#$ -pe omp 4

# Request 2 GPU (the number of GPUs needed should be divided by the number of CPUs requested above)
#$ -l gpus=0.5

# Specify the minimum GPU compute capability 
#$ -l gpu_c=6.0

# Force the job to run only on a buyin node
#$ -l buyin

# when to send email report
#$ -m bae

module load python3/3.6.9
module load gcc/5.5.0
module load cuda/10.0
module load tensorflow/2.0.0
module load opencv

# pip install --user opencv-python
# pip install --user pycocotools

python -V
python train.py

# qrsh -P ec720prj -pe omp 4 -l gpus=0.25 -l gpu_c=6.0
