#!/bin/bash
#SBATCH --job-name='PavicHDR'
#SBATCH --chdir=/home/urso/PavicHDR/logs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=7-24:00
#SBATCH --exclusive
#SBATCH --gres=gpu:4

export SLURM_OVERLAP=yes


source '/opt/intel/oneapi/setvars.sh' 
export CONDA_ENV=pytorch-gpu 



module try-load singularity
echo job start time is `date`


exec_node=${SLURM_JOB_NODELIST}
cpu_curr_node=${SLURM_CPUS_ON_NODE}
if [ "${cpu_curr_node}" = "" ]; then  cpu_curr_node={{ cores_per_node|default:1 }}; fi

set -x
srun -N1 -n1 --cpu_bind=cores -l --nodelist=${exec_node} \
    --cpus-per-task=${cpu_curr_node} \
    singularity exec --nv \
    -B /home/urso -B /scratch \
    --pwd /home/urso/PavicHDR/logs \
    /home/lico/container/pytorch.image bash /home/urso/PavicHDR/train.sh 
set +x


echo job end time is `date`
