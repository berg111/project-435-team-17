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

    double worker_receive_time,       /* Buffer for worker recieve times */
    worker_calculation_time,      /* Buffer for worker calculation times */
    worker_send_time = 0;         /* Buffer for worker send times */
    double whole_computation_time,    /* Buffer for whole computation time */
    master_initialization_time,   /* Buffer for master initialization time */
    master_send_receive_time = 0; /* Buffer for master send and receive time */

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
    if(taskid > 0){
        MPI_Comm_split(MPI_COMM_WORLD, 1, taskid, &new_comm);
    }
    else if(taskid == 0){
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, taskid, &new_comm);
    }   

    double whole_start_time = MPI_Wtime();
    double recv_total_time, send_total_time, calc_total_time = 0;

    if (taskid == MASTER) {

        generate_array(array, size, arrayINput);

        // printf("Unsorted Array: ");
        // for (int i = 0; i < size; i++) {
        //     printf("%d ", array[i]);
        // }
        mtype = FROM_MASTER;
        for (dest = 1; dest <= numworkers; dest++) {
            MPI_Send(&size, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(array, size, MPI_INT, dest, mtype, MPI_COMM_WORLD);
}

        CALI_MARK_BEGIN("comp");
        oddEvenSort(array, size);
                
        
        mtype = FROM_WORKER;
        for (int i = 1; i <= numworkers; i++) {
        source = i;
        MPI_Recv(array, size, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        }

        CALI_MARK_END("comp");

        
        // printf("Sorted Array: ");
        // for (int i = 0; i < size; i++) {
        // printf("%d ", array[i]);
        // }
        printf("\n");
    } if(taskid > MASTER) {

        mtype = FROM_MASTER;
        MPI_Recv(&size, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(array, size, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        double calc_start_time = MPI_Wtime();
        oddEvenSort(array, size);
        double calc_end_time = MPI_Wtime();
        calc_total_time = calc_end_time - calc_start_time;
        mtype = FROM_WORKER;
        MPI_Send(array, size, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    }

    double whole_end_time = MPI_Wtime();
    whole_computation_time = whole_end_time - whole_start_time;

       double worker_receive_time_max,
      worker_receive_time_min,
      worker_receive_time_sum,
      worker_recieve_time_average,
      worker_calculation_time_max,
      worker_calculation_time_min,
      worker_calculation_time_sum,
      worker_calculation_time_average,
      worker_send_time_max,
      worker_send_time_min,
      worker_send_time_sum,
      worker_send_time_average = 0; // Worker statistic values.

      if(new_comm != MPI_COMM_NULL){

        MPI_Reduce(&calc_total_time, &worker_calculation_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, new_comm);
        MPI_Reduce(&calc_total_time, &worker_calculation_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, new_comm);
        MPI_Reduce(&calc_total_time, &worker_calculation_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, new_comm);
        worker_calculation_time_average = worker_calculation_time_sum / numworkers;
      }


       if (taskid == 0)
   {
      // Master Times
      printf("******************************************************\n");
      printf("Master Times:\n");
      printf("Whole Computation Time: %f \n", whole_computation_time);
      printf("\n******************************************************\n");

      // Add values to Adiak
      adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);

      // Must move values to master for adiak
      mtype = FROM_WORKER;

      MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);


      adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
      adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
      adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
   }
   else if (taskid == 1)
   { // Print only from the first worker.
      // Print out worker time results.
      
      // Compute averages after MPI_Reduce
      worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;


      printf("******************************************************\n");
      printf("Worker Times:\n");
      printf("Worker Calculation Time Max: %f \n", worker_calculation_time_max);
      printf("Worker Calculation Time Min: %f \n", worker_calculation_time_min);
      printf("Worker Calculation Time Average: %f \n", worker_calculation_time_average);
      printf("\n******************************************************\n");

      mtype = FROM_WORKER;
      MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_calculation_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }

    MPI_Finalize();
    return 0;
}
