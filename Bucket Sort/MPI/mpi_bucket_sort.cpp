#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <iostream>
#include <string.h>
#include <random>
#include <math.h>
#include <vector>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */
#define SIZE 10                /* array size */

using std::cout, std::endl, std::string, std::vector;

void initializeData(int numDecimalPlaces, float(*array)[SIZE]) 
{
    long long int randMax = pow(10, numDecimalPlaces);

    for (int i = 0; i < SIZE; i++) {
        (*array)[i] = (float)((rand() % (randMax - 1)) + 1) / (float)randMax;
    }
}

void insertionSort(vector<float, std::allocator<float>> *array) 
{
    int index;
    float temp;

    for (int i = 1; i < (*array).size(); i++) {
        // Set index
        index = i;

        // Loop through elements behind current element
        while ((*array)[index] < (*array)[index - 1]) {
            // Swap elements
            temp = (*array)[index];
            (*array)[index] = (*array)[index - 1];
            (*array)[index - 1] = temp;

            // Change index
            index--;
            
            // End condition
            if (index == 0) {
                break;
            }
        }
    }
}

bool correctnessCheck(float (*array)[SIZE]) 
{
    for (int i = 1; i < SIZE; i++) {
        if ((*array)[i] < (*array)[i - 1]) {
            return false;
        }
    }

    return true;
}

int main (int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
    
    int numDecimals;

    if (argc == 2) {
        numDecimals = atoi(argv[1]);
    }
    else {
        printf("\n Please provide the size of the matrix");
        return 0;
    }

    int	numtasks,              /* number of tasks in partition */
	    taskid,                /* a task identifier */
	    numworkers,            /* number of worker tasks */
	    source,                /* task id of message source */
	    dest,                  /* task id of message destination */
	    mtype,                 /* message type */
	    i, j, rc;           /* misc */

    float array[SIZE];

    MPI_Status status;

    double worker_receive_time,       /* Buffer for worker recieve times */
        worker_calculation_time,      /* Buffer for worker calculation times */
        worker_send_time = 0;         /* Buffer for worker send times */

    double main_time,    /* Buffer for whole computation time */
        data_init_time,   /* Buffer for master initialization time */
        master_send_receive_time = 0; /* Buffer for master send and receive time */

    double correctness_check_time = 0;

    bool sortedTest = false;
        
    /* Define Caliper region names */
    const char* main = "main";
    const char* data_init = "data_init";
    const char* master_send_receive = "master_send_receive";
    const char* worker_receive = "worker_receive";
    const char* worker_calculation = "worker_calculation";
    const char* worker_send = "worker_send";
    const char* correctness_check = "correctness_check";

    // Initialize MPI program
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    numworkers = numtasks-1;

    // NEW COMMUNICATOR
    MPI_Comm newcomm;
    MPI_Comm_split(MPI_COMM_WORLD, taskid > 0, taskid - 1, *newcomm);

    // WHOLE PROGRAM COMPUTATION PART STARTS HERE
    CALI_MARK_BEGIN(main);
    main_time = MPI_Wtime();

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

/**************************** master task ************************************/
    if (taskid == MASTER)
    {
   
        // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE

        printf("mpi_mm has started with %d tasks.\n",numtasks);
        printf("Initializing array and buckets...\n");

        CALI_MARK_BEGIN(data_init); // Don't time printf
        data_init_time = MPI_Wtime();

        initializeData(numDecimals, &array);

        // Sets index and buckets vector
        int index = 0;
        vector<float> buckets[SIZE];

        // Pushes array elements into buckets
        for (i = 0; i < SIZE; i++) {
            buckets[(int)(SIZE * (*array)[i])].push_back((*array)[i]);
        }
            
        //INITIALIZATION PART FOR THE MASTER PROCESS ENDS HERE
        CALI_MARK_END(data_init); // Ends Caliper measurement
        data_init_time = MPI_Wtime() - data_init_time;
        
        //SEND-RECEIVE PART FOR THE MASTER PROCESS STARTS HERE
        CALI_MARK_BEGIN(master_send_receive);
        master_send_receive_time = MPI_Wtime();

        /* Send matrix data to the worker tasks */
        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++) {
            printf("Sending bucket #%d to task %d\n", dest, dest);
            MPI_Send(&(buckets[dest-1]), 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        }


        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (source = 1; source<=numworkers; source++) {
            MPI_Recv(&(buckets[source-1]), 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n",source);
        }
        
        //SEND-RECEIVE PART FOR THE MASTER PROCESS ENDS HERE
        CALI_MARK_END(master_send_receive);
        master_send_receive_time = MPI_Wtime() - master_send_receive_time;

        // Stitches each bucket back together
        for (i = 0; i < SIZE; i++) {
            for (j = 0; j < buckets[i].size(); j++) {
                (*array)[index] = buckets[i][j];
                index++;
            }
        }

        CALI_MARK_BEGIN(correctness_check);
        correctness_check_time = MPI_Wtime();

        sortedTest = correctnessCheck(&array);

        CALI_MARK_END(correctness_check);
        correctness_check_time = MPI_Wtime() - correctness_check_time;

        if (sortedTest) {
            printf("Array was correctly sorted.");
        } else {
            printf("Array was sorted incorrectly.");
        }
    }


/**************************** worker task ************************************/
    if (taskid > MASTER)
    {
        vector<float> bucket;
        bucket.push_back(0.0);
        
        //RECEIVING PART FOR WORKER PROCESS STARTS HERE
        CALI_MARK_BEGIN(worker_receive);
        worker_receive_time = MPI_Wtime();

        mtype = FROM_MASTER;
        MPI_Recv(&(bucket[0]), 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        
        //RECEIVING PART FOR WORKER PROCESS ENDS HERE
        CALI_MARK_END(worker_receive);
        worker_receive_time = MPI_Wtime() - worker_receive_time;

        //CALCULATION PART FOR WORKER PROCESS STARTS HERE
        CALI_MARK_BEGIN(worker_calculation);
        worker_calculation_time = MPI_Wtime();

        insertionSort(&(bucket[0]));
            
        //CALCULATION PART FOR WORKER PROCESS ENDS HERE
        CALI_MARK_END(worker_calculation);
        worker_calculation_time = MPI_Wtime() - worker_calculation_time;
        
        //SENDING PART FOR WORKER PROCESS STARTS HERE
        CALI_MARK_BEGIN(worker_send);
        worker_send_time = MPI_Wtime();


        mtype = FROM_WORKER;
        MPI_Send(&(bucket[0]), 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD)

        //SENDING PART FOR WORKER PROCESS ENDS HERE
        CALI_MARK_END(worker_send);
        worker_send_time = MPI_Wtime() - worker_send_time;
    }

    // WHOLE PROGRAM COMPUTATION PART ENDS HERE
    CALI_MARK_END(main);
    main_time = MPI_Wtime() - main_time;

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_procs", numtasks);
    adiak::value("num_decimals", numDecimals);
    adiak::value("program_name", "master_worker_bucket_sort");
    adiak::value("array_datatype_size", sizeof(float));

    double worker_receive_time_max,
        worker_receive_time_min,
        worker_receive_time_sum,
        worker_receive_time_average,
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
    
    // sendbuf = MPI_Wtime() counters
    // recvbuf = new vars
    // datatype -> MPI_DOUBLE
    // op -> MPI_MAX, MPI_MIN, MPI_SUM
    // There should be 9 total Reduce calls
    
    MPI_Reduce(&worker_receive_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
    MPI_Reduce(&worker_receive_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
    MPI_Reduce(&worker_receive_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);
    
    MPI_Reduce(&worker_calculation_time, &worker_calculation_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
    MPI_Reduce(&worker_calculation_time, &worker_calculation_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
    MPI_Reduce(&worker_calculation_time, &worker_calculation_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);
    
    MPI_Reduce(&worker_send_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
    MPI_Reduce(&worker_send_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
    MPI_Reduce(&worker_send_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);
    
    
    if (taskid == 0)
    {
        // Master Times
        printf("******************************************************\n");
        printf("Master Times:\n");
        printf("Main Time: %f \n", main_time);
        printf("Data Initialization Time: %f \n", data_init_time);
        printf("Master Send and Receive Time: %f \n", master_send_receive_time);
        printf("Master Correctness Check Time: %f \n", correctness_check_time)
        printf("\n******************************************************\n");

        // Add values to Adiak
        adiak::value("MPI_Reduce-main_time", main_time);
        adiak::value("MPI_Reduce-data_init_time", data_init_time);
        adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);
        adiak::value("MPI_Reduce-correctness_check_time", correctness_check_time);

        // Must move values to master for adiak
        mtype = FROM_WORKER;
        MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_receive_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

        adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
        adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
        adiak::value("MPI_Reduce-worker_receive_time_average", worker_receive_time_average);
        adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
        adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
        adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
        adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
        adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
        adiak::value("MPI_Reduce-worker_send_time_average", worker_send_time_average);
        adiak::value("sorted", sortedTest);
    }
    else if (taskid == 1)
    { // Print only from the first worker.
        // Print out worker time results.
        
        // Compute averages after MPI_Reduce
        worker_receive_time_average = worker_receive_time_sum / (double)numworkers;
        worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
        worker_send_time_average = worker_send_time_sum / (double)numworkers;

        printf("******************************************************\n");
        printf("Worker Times:\n");
        printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
        printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
        printf("Worker Receive Time Average: %f \n", worker_receive_time_average);
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
        MPI_Send(&worker_receive_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
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
}