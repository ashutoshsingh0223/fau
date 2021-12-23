#!/bin/sh
#SBATCH -N 1 -n 1
#SBATCH --time=02:00:00 
module load openmpi

mpirun -np 1 ./main 128 100 0.005
mpirun -np 1 ./main 256 100 0.005
mpirun -np 1 ./main 512 100 0.005
mpirun -np 1 ./main 1024 100 0.005



#mpirun -np 8 ./main 128 100 0.005
#mpirun -np 8 ./main 256 100 0.005
#mpirun -np 8 ./main 512 100 0.005
#mpirun -np 8 ./main 1024 100 0.005

#mpirun -np 16 ./main 128 100 0.005
#mpirun -np 16 ./main 256 100 0.005
#mpirun -np 16 ./main 512 100 0.005
#mpirun -np 16 ./main 1024 100 0.005
