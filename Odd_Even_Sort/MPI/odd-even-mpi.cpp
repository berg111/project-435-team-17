#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>


#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0           /* taskid of the first task */
#define FROM_MASTER 1      /* setting a message type for sending data to workers */
#define FROM_WORKER 2      /* setting a message type for receiving data from master */


void generate_array(int array[], int size, int input) {
    switch(input){
        case 1:
            for (int i = 0; i < size; i++) {
                array[i] = rand() % 10000;
            }
            break;
        case 2:
            for (int i = 0; i < size; i++) {
                array[i] = i;
            }
            break;
        case 3:
            int temp[size];
            for (int i = 0; i < size; i++) {
                temp[i] = i;
            }
            for(int i = 0; i < size; ++i){
                array[i] = temp[size - 1 - i];
            }
            break;
    }

}

bool isSorted(int array[], int size){
    for(int i = 0; i < size; ++i){
        if(array[i] > array[i + 1]){
            return false;
        }
    }
    return true;
}


void oddEvenSort(int array[], int size) {
    int phase, i, temp;

    for (phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) {
            for (i = 1; i < size; i += 2) {
                if (array[i] < array[i - 1]) {
                    temp = array[i];
                    array[i] = array[i - 1];
                    array[i - 1] = temp;
                }
            }
        } else {
            for (i = 1; i < size - 1; i += 2) {
                if (array[i] > array[i + 1]) {
                    temp = array[i];
                    array[i] = array[i + 1];
                    array[i + 1] = temp;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    int numtasks, taskid, numworkers, source, dest, mtype;
    int size = atoi(argv[1]);
    int arrayINput = atoi(argv[2]);
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

    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, task_id > 0, task_id - 1, &newcomm);

    double whole_start_time = MPI_Wtime();
    double recv_total_time, send_total_time, calc_total_time = 0;



    MPI_Finalize();
    return 0;
}
