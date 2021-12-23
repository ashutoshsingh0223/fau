#!/bin/sh
#SBATCH -N 2 --exclusive
module load openmpi
mpirun -np 32 ./main 128 100 0.005
mpirun -np 32 ./main 256 100 0.005
mpirun -np 32 ./main 512 100 0.005
mpirun -np 32 ./main 1024 100 0.005
