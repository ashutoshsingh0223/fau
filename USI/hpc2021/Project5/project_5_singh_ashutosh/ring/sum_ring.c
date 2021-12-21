/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS-USI Summer School.    *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS/USI take no responsibility for the use of the enclosed  *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Parallel sum using a ping-pong                      *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

#define MPI_ERR_CHECK(call) {                        \
    do { int err = call; if (err != MPI_SUCCESS) {   \
        char errstr[MPI_MAX_ERROR_STRING];           \
        int szerrstr;                                \
        MPI_Error_string(err, errstr, &szerrstr);    \
        fprintf(stderr, "MPI error at %s:%i : %s\n", \
            __FILE__, __LINE__, errstr);             \
        MPI_Abort(MPI_COMM_WORLD, 1);                \
    }} while (0);                                    \
}

#define to_right 201

int max(int num1, int num2)
{
    return (num1 > num2 ) ? num1 : num2;
}


int main (int argc, char *argv[])
{
    int my_rank, size;
    int snd_buf, rcv_buf;
    int right, left;
    int sum, i;

    MPI_Status  status;
    MPI_Request request;


    MPI_ERR_CHECK(MPI_Init(&argc, &argv));

    MPI_ERR_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));

    MPI_ERR_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));


    right = (my_rank + 1)        % size;/* get rank of neighbor to your right */
    left  = (my_rank - 1 + size) % size;/* get rank of neighbor to your left */

    /* Implement ring addition code
     * do not use if (rank == 0) .. else ..
     * every rank sends initialy its rank number to a neighbor, and then sends what
     * it receives from that neighbor, this is done n times with n = number of processes
     * all ranks will obtain the sum.
     */
    sum = 0;
    for (int i = 0, sent_buf = 3 * my_rank % (2 * size), recv_buf; i < size; i++, sent_buf = recv_buf)
	{
        MPI_Request request;
		MPI_ERR_CHECK(MPI_Isend(&sent_buf, 1, MPI_INT, right, to_right, MPI_COMM_WORLD, &request));

		MPI_Status status;
		MPI_ERR_CHECK(MPI_Recv(&recv_buf, 1, MPI_INT, left, to_right, MPI_COMM_WORLD, &status));

		MPI_ERR_CHECK(MPI_Wait(&request, &status));

		sum = max(recv_buf, sum);
    }
    printf ("Process %i:\tSum = %i\n", my_rank, sum);

    MPI_Finalize();
    return 0;
}
