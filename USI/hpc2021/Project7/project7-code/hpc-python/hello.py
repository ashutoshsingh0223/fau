from mpi4py import MPI

# get comm, size & rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

right = (rank + 1)        % size;/* get rank of neighbor to your right */
left  = (rank - 1 + size) % size;/

