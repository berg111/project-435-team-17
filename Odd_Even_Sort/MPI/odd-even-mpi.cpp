#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>


#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2         /* setting a message type */

//Helper function to create array of random values
void generate_array(int *array[], int size){
    for(int i = 0; i < size; i++){
		*array[i] = rand() % 10000;
    }
}

int oddEvenSort(int array[], int size){

    for (int phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) {
            for (i = 1; i < size; i += 2) {
                if (array[i] > array[i + 1]) {
                    std::swap(array[i], array[i + 1]);
            }
        } else {
            for (i = 1; i < size - 1; i += 2) {
                if (array[i] > array[i + 1]) {
                    std::swap(array[i], array[i + 1]);
                }
            }
        }
    }
}
}

int main(int argc, char** argv){
    int  numtasks,           /* number of tasks in partition */
	taskid,  
    numworkers,              /* a task identifier */  
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,/* message type */
    rank;

    int size = atoi(argv[1]);

    int array[size];

    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    numworkers = numtasks-1;

    MPI_Comm new_comm;
    if(taskid > 0){
        MPI_Comm_split(MPI_COMM_WORLD, 1, taskid, &new_comm);
    }
    else if(taskid == 0){
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, taskid, &new_comm);
    }

    if(taskid == MASTER){

        generate_array(&array);
        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++) {
            printf("Sending bucket #%d to task %d\n", dest, dest);
            MPI_Send(&array, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        }

        mtype = FROM_WORKER;
         for (i=1; i<=numworkers; i++)
        {
         source = i;
         MPI_Recv(&array, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
      }
    }

    if(taskid > MASTER){

        mtype = FROM_MASTER;
        MPI_Recv(&array, 1, MPI_INT, MASTER, mtype, new_comm, &status);

        oddEvenSort(array, size);

        mtype = FROM_WORKER;
        MPI_Send(&array, 1, MPI_INT, source, mtype, new_comm, &status);

        
    }

    MPI_Finalize();


}