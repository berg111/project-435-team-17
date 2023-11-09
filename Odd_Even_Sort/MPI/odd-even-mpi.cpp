#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


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

    CALI_CXX_MARK_FUNCTION;

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

    double worker_receive_time,       /* Buffer for worker recieve times */
   worker_calculation_time,      /* Buffer for worker calculation times */
   worker_send_time = 0;         /* Buffer for worker send times */
    double whole_computation_time,    /* Buffer for whole computation time */
   master_initialization_time,   /* Buffer for master initialization time */
   master_send_receive_time = 0; /* Buffer for master send and receive time */

    /* Define Caliper region names */
    const char* whole_computation = "whole_computation";
    const char* master_initialization = "master_initialization";
    const char* master_send_recieve = "master_send_recieve";
    const char* worker_recieve = "worker_recieve";
    const char* worker_calculation = "worker_calculation";
    const char* worker_send = "worker_send";

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

    CALI_MARK_BEGIN(whole_computation);

    if(taskid == MASTER){

        CALI_MARK_BEGIN(master_initialization);

        generate_array(&array);

        CALI_MARK_END(master_initialization);

        CALI_MARK_BEGIN(master_send_recieve);

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

        CALI_MARK_END(master_send_recieve);

    }

    if(taskid > MASTER){

        CALI_MARK_BEGIN(worker_recieve);

        mtype = FROM_MASTER;
        MPI_Recv(&array, 1, MPI_INT, MASTER, mtype, new_comm, &status);

        CALI_MARK_END(worker_recieve);

        CALI_MARK_BEGIN(worker_calculation);

        oddEvenSort(array, size);

        CALI_MARK_END(worker_calculation);

        CALI_MARK_BEGIN(worker_send);

        mtype = FROM_WORKER;
        MPI_Send(&array, 1, MPI_INT, source, mtype, new_comm, &status);

        CALI_MARK_END(worker_send);

        
    }

    CALI_MARK_END(whole_computation);

       adiak::init(NULL);
   adiak::user();
   adiak::launchdate();
   adiak::libraries();
   adiak::cmdline();
   adiak::clustername();
   adiak::value("num_procs", numtasks);
   adiak::value("matrix_size", sizeOfMatrix);
   adiak::value("program_name", "master_worker_matrix_multiplication");
   adiak::value("matrix_datatype_size", sizeof(double));

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

   /* USE MPI_Reduce here to calculate the minimum, maximum and the average times for the worker processes.
   MPI_Reduce (&sendbuf,&recvbuf,count,datatype,op,root,comm). https://hpc-tutorials.llnl.gov/mpi/collective_communication_routines/ */
   if(new_comm != MPI_COMM_NULL){

      MPI_Reduce(&send_total_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, new_comm);
      MPI_Reduce(&send_total_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, new_comm);
      MPI_Reduce(&send_total_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, new_comm);
      worker_send_time_average = worker_send_time_sum / numworkers;
      MPI_Reduce(&recv_total_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, new_comm);
      MPI_Reduce(&recv_total_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, new_comm);
      MPI_Reduce(&recv_total_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, new_comm);
      worker_recieve_time_average = worker_send_time_sum / numworkers;
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
      printf("Master Initialization Time: %f \n", master_initialization_time);
      printf("Master Send and Receive Time: %f \n", master_send_receive_time);
      printf("\n******************************************************\n");

      // Add values to Adiak
      adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);
      adiak::value("MPI_Reduce-master_initialization_time", master_initialization_time);
      adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);

      // Must move values to master for adiak
      mtype = FROM_WORKER;
      MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_recieve_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_send_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

      adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
      adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
      adiak::value("MPI_Reduce-worker_recieve_time_average", worker_recieve_time_average);
      adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
      adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
      adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
      adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
      adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
      adiak::value("MPI_Reduce-worker_send_time_average", worker_send_time_average);
   }
   else if (taskid == 1)
   { // Print only from the first worker.
      // Print out worker time results.
      
      // Compute averages after MPI_Reduce
      worker_recieve_time_average = worker_receive_time_sum / (double)numworkers;
      worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
      worker_send_time_average = worker_send_time_sum / (double)numworkers;

      printf("******************************************************\n");
      printf("Worker Times:\n");
      printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
      printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
      printf("Worker Receive Time Average: %f \n", worker_recieve_time_average);
      printf("Worker Calculation Time Max: %f \n", worker_calculation_time_max);
      printf("Worker Calculation Time Min: %f \n", worker_calculation_time_min);
      printf("Worker Calculation Time Average: %f \n", worker_calculation_time_average);
      printf("Worker Send Time Max: %f \n", worker_send_time_max);
      printf("Worker Send Time Min: %f \n", worker_send_time_min);
      printf("Worker Send Time Average: %f \n", worker_send_time_average);
      printf("\n******************************************************\n");

      mtype = FROM_WORKER;
      MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_recieve_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_calculation_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_send_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }

   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();

    MPI_Finalize();


}