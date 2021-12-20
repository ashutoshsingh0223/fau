from mpi4py import MPI
import sys
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

method = 'direct'
if len(sys.argv) > 1:
    method = str(sys.argv[1])

if method == 'pickle':
    sum = 0
    if rank == 0:
        for j in range(1, size):
            receive_buf = comm.recv(None, source=j, tag=42)
            sum += receive_buf

    else:
        comm.send(rank, dest=0, tag=42)

    if rank == 0:
        print(f"Pickle. Sum of all ranks for n={size} is {sum}")

else:
    sum = 0
    if rank == 0:
        for j in range(1, size):
            receive_buf = np.empty(1, dtype='i')
            comm.Recv([receive_buf, MPI.INT], source=j, tag=42)
            sum += receive_buf[0]

    else:
        comm.Send([np.array([rank], dtype='i'), MPI.INT], dest=0, tag=42)

    if rank == 0:
        print(f"Direct Array Data. Sum of all ranks for n={size} is {sum}")







