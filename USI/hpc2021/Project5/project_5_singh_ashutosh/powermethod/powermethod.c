/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS-USI Summer School.    *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS/USI take no responsibility for the use of the enclosed  *
 * teaching material.                                           *
 *                                                              *
 * Purpose: : Parallel matrix-vector multiplication and the     *
 *            and power method                                  *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#include "hpc-power.h"


double norm(double* x, int n){
    /* 
    This method calculates the l2-norm of a vector `x` given its size `n`
    arguments:
    x = the vector to calculate norm of 
    n = size of the vector

    returns:
    the value of norm of the vector
    */

    double result = 0;
    for(int i = 0; i < n; i++){
        result += x[i] * x[i];
    }
    result = sqrt(result);
    return result;
}


void matVec(double* A, double* x, double* result, int n, int n_rows){

    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);    

    double y[n_rows];

    for(int i = 0; i < n_rows; i++){
        y[i] = 0;
        for (int j = 0; j < n; j++){
            y[i] += A[i * n + j] * x[j];
        }
    }
    
    MPI_Gather(y, n_rows, MPI_DOUBLE, result, n_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}



double powerMethod(double* A, int n, int iterations, int n_rows){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Vector x
    double x[n];
    if( rank == 0 ){
        for(int i=0; i < n; i++){
            x[i]=1;
        }
    }


    // Power Method
    for(int iter=0; iter < iterations; iter++){
        // Normalize
        if( rank == 0 ){
            double norm2 = norm(x, n);
            for(int i = 0; i < n; i++){
                x[i] = x[i]/norm2;
            }
        }


        // matrix multiplication
        double temp[n];
        matVec(A, x, temp, n, n_rows);
        if( rank == 0 ){
            for(int i = 0; i < n; i++){
                x[i] = temp[i];
            }
        }
    }
    

    if( rank == 0 ){
        int correct = hpc_verify(x, n, 0);
        printf("Correct - %d      ", correct);

    }


    return norm(x, n);
}


double* generateMatrix(int n, int startrow, int numrows) {
    /*Generates a slice of matrix A
        arguments:
        n = the number of columns (and rows) in A
        startrow = the row to start on(given as =  rank * n/p)
        numrows = the number of rows to generate( given as  = n/p)
    */
	double* A;
	int i;
	int diag;

	A = (double*)calloc(n * numrows, sizeof(double));

	for (i = 0; i < numrows; i++) {
		diag = startrow + i;

		A[i * n + diag] = n;
	}

	return A;
}


int nForWeakScaling(int size, int n){
    int p_root = sqrt(size);
    n = n * p_root;
    return n;
}   


int main (int argc, char *argv[])
{
    int my_rank, size;
    int snd_buf, rcv_buf;
    int right, left;
    int sum, i;


    MPI_Status  status;
    MPI_Request request;


    double* A;
    double lambda;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10000;
    int iterations = 1000;

    // Uncomment this line when reproducing for weak scaling
    // n = nForWeakScaling(size, n);


    A = generateMatrix(n, my_rank * n/size, n/size);

    double startTime = hpc_timer();

    lambda = powerMethod(A, n, 1000, n/size);

    
    double endTime = hpc_timer();

    if( my_rank == 0 ){
        printf("%.6f     ", endTime - startTime);
        printf("%d\n", n);
    }


    /* This subproject is about to write a parallel program to
       multiply a matrix A by a vector x, and to use this routine in
       an implementation of the power method to find the absolute
       value of the largest eigenvalue of the matrix. Your code will
       call routines that we supply to generate matrices, record
       timings, and validate the answer.
    */

    MPI_Finalize();
    return 0;
}
