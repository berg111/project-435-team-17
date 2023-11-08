#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>

int main(int argc, char** argv){
    int             /* number of tasks in partition */
	taskid,                /* a task identifier */  
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
    rank;

    MPI_Status status;

    MPI_Init(&argc,&argv);
}