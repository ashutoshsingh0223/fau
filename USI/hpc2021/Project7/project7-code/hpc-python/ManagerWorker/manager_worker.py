from mandelbrot_task import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI # MPI_Init and MPI_Finalize automatically called
import numpy as np
import sys
import time
import math

# some parameters
MANAGER = 0       # rank of manager
TAG_TASK      = 1 # task       message tag
TAG_TASK_DONE = 2 # tasks done message tag
TAG_DONE      = 3 # done       message tag

def manager(comm, tasks):
    """
    The manager.

    Parameters
    ----------
    comm : mpi4py.MPI communicator
        MPI communicator
    tasks : list of objects with a do_task() method perfroming the task
        List of tasks to accomplish

    Returns
    -------
    ... ToDo ...
    """
    size = comm.Get_size()
    num_workers = size  - 1

    def split(tasks, n):
        k, m = divmod(len(tasks), n)
        return (tasks[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    chunks = split(tasks, num_workers)

    for i, chunk in enumerate(chunks):
        TasksDoneByWorker[i+1] = len(chunk)
        comm.send(chunk, dest=i+1, tag=TAG_TASK)

    
    tasks_done = []
    for i in range(num_workers):
        tasks_done.extend(comm.recv(None, source=i+1, tag=TAG_TASK_DONE))
    
    return tasks_done


def worker(comm):

    """
    The worker.

    Parameters
    ----------
    comm : mpi4py.MPI communicator
        MPI communicator
    """
    my_rank = comm.Get_rank()
    tasks = comm.recv(None, source=MANAGER, tag=TAG_TASK)
    print(f'Worker-{my_rank} received {len(tasks)} tasks')
    for task in tasks:
        task.do_work()

    comm.send(tasks, dest=MANAGER, tag=TAG_TASK_DONE)

    return len(tasks)
    


def readcmdline(rank):
    """
    Read command line arguments

    Parameters
    ----------
    rank : int
        Rank of calling MPI process

    Returns
    -------
    nx : int
        number of gridpoints in x-direction
    ny : int
        number of gridpoints in y-direction
    ntasks : int
        number of tasks
    """
    # report usage
    if len(sys.argv) != 4:
        if rank == MANAGER:
            print("Usage: manager_worker nx ny ntasks")
            print("  nx     number of gridpoints in x-direction")
            print("  ny     number of gridpoints in y-direction")
            print("  ntasks number of tasks")
        sys.exit()

    # read nx, ny, ntasks
    nx = int(sys.argv[1])
    if nx < 1:
        sys.exit("nx must be a positive integer")
    ny = int(sys.argv[2])
    if ny < 1:
        sys.exit("ny must be a positive integer")
    ntasks = int(sys.argv[3])
    if ntasks < 1:
        sys.exit("ntasks must be a positive integer")

    return nx, ny, ntasks


if __name__ == "__main__":

    # get COMMON WORLD communicator, size & rank
    comm    = MPI.COMM_WORLD
    size    = comm.Get_size()
    my_rank = comm.Get_rank()

    # report on MPI environment
    if my_rank == MANAGER:
        print(f"MPI initialized with {size:5d} processes")

    # read command line arguments
    nx, ny, ntasks = readcmdline(my_rank)
    if my_rank == MANAGER:
        print(f"MPI initialized with {size:5d} processes")

    TasksDoneByWorker = [None] * (size)

    # trying out ... YOUR MANAGER-WORKER IMPLEMENTATION HERE ...
    x_min = -2.
    x_max  = +1.
    y_min  = -1.5
    y_max  = +1.5

    if my_rank == MANAGER:
        # start timer
        timespent = - time.perf_counter()
        M = mandelbrot(x_min, x_max, nx, y_min, y_max, ny, ntasks)
        tasks = M.get_tasks()
        tasks = manager(comm, tasks)
        # stop timer
        timespent += time.perf_counter()
        m = M.combine_tasks(tasks)
        plt.imshow(m.T, cmap="gray", extent=[x_min, x_max, y_min, y_max])
        plt.savefig("mandelbrot.png")
    
    else:
        num_tasks = worker(comm)
        TasksDoneByWorker[my_rank - 1] = num_tasks
        # print(TasksDoneByWorker)


    # inform that done
    if my_rank == MANAGER:
        print(f"Run took {timespent:f} seconds")
        for i in range(size):
            if i == MANAGER:
                continue
            print(f"Process {i:5d} has done {TasksDoneByWorker[i]:10d} tasks")
        print("Done.")

