/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS-USI Summer School     *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS/USI take no responsibility for the use of the enclosed  *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Exchange ghost cell in 2 directions using a topology*
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/* Use only 16 processes for this exercise
 * Send the ghost cell in two directions: left<->right and top<->bottom
 * ranks are connected in a cyclic manner, for instance, rank 0 and 12 are connected
 *
 * process decomposition on 4*4 grid
 *
 * |-----------|
 * | 0| 1| 2| 3|
 * |-----------|
 * | 4| 5| 6| 7|
 * |-----------|
 * | 8| 9|10|11|
 * |-----------|
 * |12|13|14|15|
 * |-----------|
 *
 * Each process works on a 6*6 (SUBDOMAIN) block of data
 * the D corresponds to data, g corresponds to "ghost cells"
 * xggggggggggx
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * xggggggggggx
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SUBDOMAIN 6
#define DOMAINSIZE (SUBDOMAIN+2)

#define to_right 201

int main(int argc, char *argv[])
{
    int rank, size, i, j, dims[2], periods[2], rank_top, rank_bottom, rank_left, rank_right;
    int rank_cart;
    int coords[2];
    double data[DOMAINSIZE*DOMAINSIZE];
    MPI_Request request;
    MPI_Status status;
    MPI_Comm comm_cart;
    MPI_Comm comm;
    MPI_Datatype data_ghost;

    comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size!=16) {
        printf("please run this with 16 processors\n");
        MPI_Finalize();
        exit(1);
    }

    // initialize the domain
    for (i=0; i<DOMAINSIZE*DOMAINSIZE; i++) {
        data[i]=rank;
    }

    // TODO: set the dimensions of the processor grid and periodic boundaries in both dimensions
    dims[0] = 4; 
    dims[1] = 4;
    periods[0] = 1;
    periods[1] = 1;

    // TODO: Create a Cartesian communicator (4*4) with periodic boundaries (we do not allow
    // the reordering of ranks) and use it to find your neighboring
    // ranks in all dimensions in a cyclic manner.
    MPI_Cart_create(comm, 2, dims, periods, 0, &comm_cart);


    
    // TODO: find your top/bottom/left/right neighbor using the new communicator, see MPI_Cart_shift()
    // rank_top, rank_bottom
    // rank_left, rank_right

    MPI_Cart_shift(comm_cart, 1, +1, &rank_left, &rank_right);
    // printf("CurrentProcess=%d, source=%d, dest=%d\n", rank, rank_left, rank_right);
    MPI_Cart_shift(comm_cart, 0, -1, &rank_bottom, &rank_top);
    // printf("CurrentProcess=%d, source=%d, dest=%d\n", rank, rank_bottom, rank_top);

    //  TODO: create derived datatype data_ghost, create a datatype for sending the column, see MPI_Type_vector() and MPI_Type_commit()
    // data_ghost
    
    // After every stride , pickup one 1 block of memory(for that datatype) for a total of SUBDOMAIN blocks
    MPI_Type_vector(SUBDOMAIN, 1, DOMAINSIZE, MPI_DOUBLE, &data_ghost);
    MPI_Type_commit(&data_ghost);


    //  TODO: ghost cell exchange with the neighbouring cells in all directions
    //  use MPI_Irecv(), MPI_Send(), MPI_Wait() or other viable alternatives

    double recv_data[SUBDOMAIN];
    //  to the top
    MPI_Isend(&data[1], SUBDOMAIN, MPI_DOUBLE, rank_top, to_right, comm_cart, &request);
    MPI_Wait(&request, &status);

    //  to the bottom
    MPI_Isend(&data[DOMAINSIZE * DOMAINSIZE - 1 - SUBDOMAIN], SUBDOMAIN, MPI_DOUBLE, rank_bottom, to_right, comm_cart, &request);
    MPI_Wait(&request, &status);

    //  to the left
    MPI_Isend(&data, 1, data_ghost, rank_left, to_right, comm_cart, &request);
    MPI_Wait(&request, &status);
    
    // //  to the right
    MPI_Isend(&data, 1, data_ghost, rank_right, to_right, comm_cart, &request);
    MPI_Wait(&request, &status);



    MPI_Irecv(&data[1], SUBDOMAIN, MPI_DOUBLE, rank_top, to_right, comm_cart, &request);
    MPI_Wait(&request, &status);
    MPI_Irecv(&data[DOMAINSIZE * DOMAINSIZE - 1 - SUBDOMAIN], SUBDOMAIN, MPI_DOUBLE, rank_bottom, to_right, comm_cart, &request);
    MPI_Wait(&request, &status);

    MPI_Irecv(recv_data, SUBDOMAIN, MPI_DOUBLE, rank_left, to_right, comm_cart, &request);
    MPI_Wait(&request, &status);

    for (j=1; j<=SUBDOMAIN; j++){
        data[j * DOMAINSIZE] = recv_data[j - 1];
    }


    MPI_Irecv(recv_data, SUBDOMAIN, MPI_DOUBLE, rank_right, to_right, comm_cart, &request);
    MPI_Wait(&request, &status);

    for (j=1; j<=SUBDOMAIN; j++){
        data[(j * DOMAINSIZE) + DOMAINSIZE - 1] = recv_data[j - 1];
    }



    if (rank==9) {
        printf("data of rank 9 after communication\n");
        for (j=0; j<DOMAINSIZE; j++) {
            for (i=0; i<DOMAINSIZE; i++) {
                printf("%.1f ", data[i+j*DOMAINSIZE]);
            }
            printf("\n");
        }
    }

    // TODO: uncomment when done with tasks above
    MPI_Type_free(&data_ghost);
    MPI_Comm_free(&comm_cart);
    MPI_Finalize();

    return 0;
}
