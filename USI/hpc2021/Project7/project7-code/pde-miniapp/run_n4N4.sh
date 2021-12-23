#!/bin/sh
#SBATCH -N 4 -n 4
#SBATCH --time=02:00:00
module load openmpi
mpirun -np 4 ./main 128 100 0.005
mpirun -np 4 ./main 256 100 0.005
mpirun -np 4 ./main 512 100 0.005
mpirun -np 4 ./main 1024 100 0.005
