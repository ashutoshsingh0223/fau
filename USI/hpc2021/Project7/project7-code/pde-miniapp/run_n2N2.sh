#!/bin/sh
#SBATCH -N 2 -n 2
#SBATCH --time=02:00:00
module load openmpi
mpirun -np 2 ./main 128 100 0.005
mpirun -np 2 ./main 256 100 0.005
mpirun -np 2 ./main 512 100 0.005
mpirun -np 2 ./main 1024 100 0.005
