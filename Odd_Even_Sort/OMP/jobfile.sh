#!/bin/bash
### Job commands start here
## echo '=====================JOB STARTING=========================='
#SBATCH --job-name=openmpcode              ### Job Name
#SBATCH --output=output.%j        ### File in which to store job output
#SBATCH --time=0-00:05:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job, default = 1
#SBATCH --mem=8G                ### memory per node
#SBATCH --exclusive             ### no shared resources within a node
#SBATCH --partition=short       ### Which hardware partition to run on
export OMP_PROC_BIND='close'
time_now=$(date +%s)
echo ' thread affinity/proc_bind = ' ; echo $OMP_PROC_BIND
# Load modules
module load intel/2020b
module load GCCcore/8.3.0
module load Clang/9.0.1
# Set variables from input
threads=$1
arraySize=$2
input=$3

CALI_USE_OMPT=1 \
CALI_CONFIG="spot(output=t${threads}-m${arraySize}-i${input}.cali, \
    time.variance, \
    openmp.threads, \
    openmp.times)" \
./odd-even $threads $arraySize $input