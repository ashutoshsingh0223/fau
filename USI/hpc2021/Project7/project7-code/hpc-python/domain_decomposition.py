from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



dims = MPI.Compute_dims(size, 2)

cart_comm = comm.Create_cart(dims, periods=[True, True])


coords = cart_comm.Get_coords(rank)
rank_left, rank_right = cart_comm.Shift(0, +1)
rank_bottom, rank_top = cart_comm.Shift(1, +1)

print(f"Process with rank={rank}. Coordinates={coords}. Neighbours: left={rank_left}. right={rank_right}, bottom={rank_bottom} and top={rank_top}")


req = cart_comm.isend(rank, dest=rank_left, tag=1)
req.wait()

req = cart_comm.isend(rank, dest=rank_right, tag=1)
req.wait()

req = cart_comm.isend(rank, dest=rank_bottom, tag=1)
req.wait()

req = cart_comm.isend(rank, dest=rank_top, tag=1)
req.wait()


received_ranks = [0]*4
req = cart_comm.irecv(source=rank_left, tag=1)
received_ranks[0] = req.wait()

req = cart_comm.irecv(source=rank_right, tag=1)
received_ranks[1] = req.wait()

req = cart_comm.irecv(source=rank_bottom, tag=1)
received_ranks[2] = req.wait()

req = cart_comm.irecv(source=rank_top, tag=1)
received_ranks[3] = req.wait()
print(f"Process with rank={rank}.Received Ranks: left={received_ranks[0]}. right={received_ranks[1]}, bottom={received_ranks[2]} and top={received_ranks[3]}")