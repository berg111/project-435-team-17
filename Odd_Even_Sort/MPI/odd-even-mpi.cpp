#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0           /* taskid of the first task */
#define FROM_MASTER 1      /* setting a message type for sending data to workers */
#define FROM_WORKER 2      /* setting a message type for receiving data from master */


void generate_array(int array[], int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 10000;
    }
}


void oddEvenSort(int localArr[], int localSize) {
    int phase, i, temp;

    for (phase = 0; phase < localSize; phase++) {
        if (phase % 2 == 0) {
            for (i = 1; i < localSize; i += 2) {
                if (localArr[i] < localArr[i - 1]) {
                    temp = localArr[i];
                    localArr[i] = localArr[i - 1];
                    localArr[i - 1] = temp;
                }
            }
        } else {
            for (i = 1; i < localSize - 1; i += 2) {
                if (localArr[i] > localArr[i + 1]) {
                    temp = localArr[i];
                    localArr[i] = localArr[i + 1];
                    localArr[i + 1] = temp;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int numtasks, taskid, numworkers, source, dest, mtype;
    int size = atoi(argv[1]);
    int array[size];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    numworkers = numtasks - 1;

    if (taskid == MASTER) {

        generate_array(array, size);
        mtype = FROM_MASTER;
        for (dest = 1; dest <= numworkers; dest++) {
            MPI_Send(&size, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(array, size, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        }

        
        oddEvenSort(array, size);

       
        mtype = FROM_WORKER;
        for (int i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(array, size, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        }

        
        // printf("Sorted Array: ");
        // for (int i = 0; i < size; i++) {
        //     printf("%d ", array[i]);
        // }
        // printf("\n");
    } else {
        
        mtype = FROM_MASTER;
        MPI_Recv(&size, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(array, size, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        
        oddEvenSort(array, size);

       
        mtype = FROM_WORKER;
        MPI_Send(array, size, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
