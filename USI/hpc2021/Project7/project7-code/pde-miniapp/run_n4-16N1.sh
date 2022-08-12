#!/bin/sh
#SBATCH -N 1 --exclusive
module load openmpi
mpirun -np 4 ./main 128 100 0.005
mpirun -np 4 ./main 256 100 0.005
mpirun -np 4 ./main 512 100 0.005
mpirun -np 4 ./main 1024 100 0.005


mpirun -np 8 ./main 128 100 0.005
mpirun -np 8 ./main 256 100 0.005
mpirun -np 8 ./main 512 100 0.005
mpirun -np 8 ./main 1024 100 0.005

mpirun -np 16 ./main 128 100 0.005
mpirun -np 16 ./main 256 100 0.005
mpirun -np 16 ./main 512 100 0.005
mpirun -np 16 ./main 1024 100 0.005