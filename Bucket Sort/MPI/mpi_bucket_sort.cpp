#include <iostream>
#include <random>
#include <math.h>
#include <vector>
#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int NUM_VALS;

size_t bytes;

void initializeData(int numDecimalPlaces, float *array, int size) 
{   
    long long int randMax = pow(10, numDecimalPlaces);

    for (int i = 0; i < size; i++) {
        array[i] = (float)((rand() % (randMax - 1)) + 1) / (float)randMax;
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

bool correctnessCheck(float *array, int size) 
{
    for (int i = 1; i < size; i++) {
        if (array[i] < array[i - 1]) {
            return false;
        }
    }

    return true;
}

void printArray(float* array, int size)
{
    for (int i = 0; i < size; i++) {
        printf("array[%d] = %f\n", i, array[i]);
    }
    printf("\n");
}


int main(int argc, char* argv[])
{
    CALI_CXX_MARK_FUNCTION;

    // Initialize number of processors and number of values
    NUM_VALS = atoi(argv[1]);

    // Initialize size of array
    bytes = NUM_VALS * sizeof(float);

    /* Define Caliper region names */
    const char* _main = "_main";
    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comm_small = "comm_small";
    const char* comm_large = "comm_large";
    const char* comp = "comp";
    const char* comp_small = "comp_small";
    const char* comp_large = "comp_large";
    const char* correctness_check = "correctness_check";

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Status status;

    // Get the task ID
    int task_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    CALI_MARK_BEGIN(_main);

    // Rank 0 coordinates work, other ranks do work
    if (task_id == 0) {
        // Get the total number of tasks
        int num_tasks;
        MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

        printf("Number of tasks: %d\n", num_tasks);
        printf("Number of values: %d\n", NUM_VALS);

        // Initialize array
        float* array = (float*) malloc(bytes);

        // Initialize data to be sorted
        printf("Initializing data...\n");
        CALI_MARK_BEGIN(data_init);
        initializeData(5, array, NUM_VALS);
        CALI_MARK_END(data_init);

        // Sets index
        int index = 0;

        // Creates buckets
        vector<float>* buckets = new std::vector<float, std::allocator<float>>[NUM_VALS];

        // Pushes array elements into buckets
        printf("Sorting into buckets...\n");
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        for (int i = 0; i < NUM_VALS; i++) {
            buckets[(int)(NUM_VALS * array[i])].push_back(array[i]);
        }
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        // Send buckets to worker tasks to be sorted w/ insertion sort
        int avgBucket = NUM_VALS/(num_tasks - 1);
        int extraBuckets = NUM_VALS%(num_tasks - 1);
        int offset = 0;
        int mtype = FROM_MASTER;
        int numBucket;
        float negative = -1.0;

        printf("Sending data...\n");
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        for (int dest=1; dest < num_tasks; dest++)
        {
            numBucket = (dest <= extraBuckets) ? avgBucket+1 : avgBucket;
            //CALI_MARK_BEGIN(comm_small);
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&numBucket, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            //CALI_MARK_END(comm_small);
            
            //CALI_MARK_BEGIN(comm_large);
            for (int i = 0; i < numBucket; i++) {
                for (int j = 0; j < buckets[i+offset].size(); j++) {
                    MPI_Send(&(buckets[i+offset].at(j)), 1, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
                }
                MPI_Send(&negative, 1, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
            }
            //CALI_MARK_END(comm_large);
            offset = offset + numBucket;
        }
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        buckets = new std::vector<float, std::allocator<float>>[NUM_VALS];

        // Receive buckets back from each thread
        int recv;
        float recvf;
        mtype = FROM_WORKER;

        printf("Receiving data...\n");
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        for (int source = 1; source < num_tasks; source++) {
            //CALI_MARK_BEGIN(comm_small);
            MPI_Recv(&recv, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            offset = recv;
            
            MPI_Recv(&recv, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            numBucket = recv;
            //CALI_MARK_END(comm_small);

            //CALI_MARK_BEGIN(comm_large);
            for (int i = 0; i < numBucket; i++) {
                MPI_Recv(&recvf, 1, MPI_FLOAT, source, mtype, MPI_COMM_WORLD, &status);
                while (recvf > 0) {
                    buckets[i + offset].push_back(recvf);
                    MPI_Recv(&recvf, 1, MPI_FLOAT, source, mtype, MPI_COMM_WORLD, &status);
                }
            }
            //CALI_MARK_END(comm_large);
        }
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        // Stitch buckets back together
        printf("Stitching buckets back together...\n");
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        for (int i = 0; i < NUM_VALS; i++) {
            for (int j = 0; j < buckets[i].size(); j++) {
                array[index] = buckets[i][j];
                index++;
            }
        }
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        // Complete correctness check
        printf("Starting correctness check...\n");
        CALI_MARK_BEGIN(correctness_check);
        bool check = correctnessCheck(array, NUM_VALS);
        CALI_MARK_END(correctness_check);
        if (check) {
            printf("Array is sorted correctly!\n");
        } else {
            printf("Array is incorrectly sorted.\n");
        }

        free(array);

        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "BucketSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
        adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", num_tasks); // The number of processors (MPI ranks)
        // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
        // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
        adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online & Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    } else {
        // Initialize variables
        int offset;
        int numBucket;
        int recv;
        float recvf;
        int mtype;
        float negative = -1.0;

        // Receive integer from rank 0
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        mtype = FROM_MASTER;
        MPI_Recv(&recv, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        offset = recv;

        MPI_Recv(&recv, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        numBucket = recv;

        vector<float>* buckets = new std::vector<float, std::allocator<float>>[numBucket];
        for (int i = 0; i < numBucket; i++) {
            MPI_Recv(&recvf, 1, MPI_FLOAT, 0, mtype, MPI_COMM_WORLD, &status);
            while (recvf > 0) {
                buckets[i].push_back(recvf);
                MPI_Recv(&recvf, 1, MPI_FLOAT, 0, mtype, MPI_COMM_WORLD, &status);
            }
        }
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);

        if (task_id == 1) {
            printf("Calculating data...\n");
        }
        
        // Do insertion sort on buckets
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        for (int i = 0; i < numBucket; i++) {
            insertionSort(&(buckets[i]));
        }
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        // Send back sorted buckets
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
        MPI_Send(&numBucket, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
        for (int i = 0; i < numBucket; i++) {
            for (int j = 0; j < buckets[i].size(); j++) {
                MPI_Send(&(buckets[i].at(j)), 1, MPI_FLOAT, 0, mtype, MPI_COMM_WORLD);
            }
            MPI_Send(&negative, 1, MPI_FLOAT, 0, mtype, MPI_COMM_WORLD);
        }
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    CALI_MARK_END(_main);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    // Finish MPI work
    MPI_Finalize();

    return 0;
}

// #include "mpi.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <limits.h>

// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>

// #include <iostream>
// #include <string.h>
// #include <random>
// #include <math.h>
// #include <vector>

// #define MASTER 0               /* taskid of first task */
// #define FROM_MASTER 1          /* setting a message type */
// #define FROM_WORKER 2          /* setting a message type */

// using std::cout, std::endl, std::string, std::vector;

// void initializeData(int numDecimalPlaces, float(*array)[SIZE]) 
// {
//     long long int randMax = pow(10, numDecimalPlaces);

//     for (int i = 0; i < SIZE; i++) {
//         (*array)[i] = (float)((rand() % (randMax - 1)) + 1) / (float)randMax;
//     }
// }

// void insertionSort(vector<float, std::allocator<float>> *array) 
// {
//     int index;
//     float temp;

//     for (int i = 1; i < (*array).size(); i++) {
//         // Set index
//         index = i;

//         // Loop through elements behind current element
//         while ((*array)[index] < (*array)[index - 1]) {
//             // Swap elements
//             temp = (*array)[index];
//             (*array)[index] = (*array)[index - 1];
//             (*array)[index - 1] = temp;

//             // Change index
//             index--;
            
//             // End condition
//             if (index == 0) {
//                 break;
//             }
//         }
//     }
// }

// bool correctnessCheck(float (*array)[SIZE]) 
// {
//     for (int i = 1; i < SIZE; i++) {
//         if ((*array)[i] < (*array)[i - 1]) {
//             return false;
//         }
//     }

//     return true;
// }

// int main (int argc, char *argv[])
// {
//     CALI_CXX_MARK_FUNCTION;
    
//     int numDecimals;

//     if (argc == 2) {
//         numDecimals = atoi(argv[1]);
//     }
//     else {
//         printf("\n Please provide the size of the matrix");
//         return 0;
//     }

//     int	numtasks,              /* number of tasks in partition */
// 	    taskid,                /* a task identifier */
// 	    numworkers,            /* number of worker tasks */
// 	    source,                /* task id of message source */
// 	    dest,                  /* task id of message destination */
// 	    mtype,                 /* message type */
// 	    i, j, rc;           /* misc */

//     float array[SIZE];

//     MPI_Status status;

//     double worker_receive_time,       /* Buffer for worker recieve times */
//         worker_calculation_time,      /* Buffer for worker calculation times */
//         worker_send_time = 0;         /* Buffer for worker send times */

//     double main_time,    /* Buffer for whole computation time */
//         data_init_time,   /* Buffer for master initialization time */
//         master_send_receive_time = 0; /* Buffer for master send and receive time */

//     double correctness_check_time = 0;
//     double comm_time, comm_small_time, comm_large_time = 0;
//     double comp_time, comp_small_time, comp_large_time = 0;

//     bool sortedTest = false;
        
    // /* Define Caliper region names */
    // const char* main = "main";
    // const char* data_init = "data_init";
    // const char* comm = "comm";
    // const char* comm_small = "comm_small";
    // const char* comm_large = "comm_large";
    // const char* comp = "comp";
    // const char* comp_small = "comp_small";
    // const char* comp_large = "comp_large";

//     const char* master_send_receive = "master_send_receive";
//     const char* worker_receive = "worker_receive";
//     const char* worker_calculation = "worker_calculation";
//     const char* worker_send = "worker_send";
//     const char* correctness_check = "correctness_check";

//     // Initialize MPI program
//     MPI_Init(&argc,&argv);
//     MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
//     MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

//     if (numtasks < 2 ) {
//         printf("Need at least two MPI tasks. Quitting...\n");
//         MPI_Abort(MPI_COMM_WORLD, rc);
//         exit(1);
//     }

//     numworkers = numtasks-1;

//     // NEW COMMUNICATOR
//     MPI_Comm newcomm;
//     MPI_Comm_split(MPI_COMM_WORLD, taskid > 0, taskid - 1, *newcomm);

//     // WHOLE PROGRAM COMPUTATION PART STARTS HERE
//     CALI_MARK_BEGIN(main);
//     main_time = MPI_Wtime();

//     // Create caliper ConfigManager object
//     cali::ConfigManager mgr;
//     mgr.start();

// /**************************** master task ************************************/
//     if (taskid == MASTER)
//     {
   
//         // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE

//         printf("mpi_mm has started with %d tasks.\n",numtasks);
//         printf("Initializing array and buckets...\n");

//         CALI_MARK_BEGIN(data_init); // Don't time printf
//         data_init_time = MPI_Wtime();

//         initializeData(numDecimals, &array);

//         // Sets index and buckets vector
//         int index = 0;
//         vector<float> buckets[SIZE];

//         // Pushes array elements into buckets
//         for (i = 0; i < SIZE; i++) {
//             buckets[(int)(SIZE * (*array)[i])].push_back((*array)[i]);
//         }
            
//         //INITIALIZATION PART FOR THE MASTER PROCESS ENDS HERE
//         CALI_MARK_END(data_init); // Ends Caliper measurement
//         data_init_time = MPI_Wtime() - data_init_time;
        
//         //SEND-RECEIVE PART FOR THE MASTER PROCESS STARTS HERE
//         CALI_MARK_BEGIN(master_send_receive);
//         master_send_receive_time = MPI_Wtime();

//         // Comm and Comm_large CALI regions start here
//         CALI_MARK_BEGIN(comm);
//         comm_time = MPI_Wtime() - comm_time;
//         CALI_MARK_BEGIN(comm_large);
//         comm_large_time = MPI_Wtime() - comm_large_time;

//         /* Send matrix data to the worker tasks */
//         mtype = FROM_MASTER;
//         for (dest=1; dest<=numworkers; dest++) {
//             printf("Sending bucket #%d to task %d\n", dest, dest);
//             MPI_Send(&(buckets[dest-1]), 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
//         }


//         /* Receive results from worker tasks */
//         mtype = FROM_WORKER;
//         for (source = 1; source<=numworkers; source++) {
//             MPI_Recv(&(buckets[source-1]), 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
//             printf("Received results from task %d\n",source);
//         }
        
//         // Comm and Comm_large CALI regions end here
//         CALI_MARK_END(comm_large);
//         comm_large_time = MPI_Wtime() - comm_large_time;
//         CALI_MARK_END(comm);
//         comm_time = MPI_Wtime() - comm_time;

//         //SEND-RECEIVE PART FOR THE MASTER PROCESS ENDS HERE
//         CALI_MARK_END(master_send_receive);
//         master_send_receive_time = MPI_Wtime() - master_send_receive_time;

//         // Comp and Comp_large CALI regions start here
//         CALI_MARK_BEGIN(comp);
//         comp_time = MPI_Wtime() - comp_time;
//         CALI_MARK_BEGIN(comp_large);
//         comp_large_time = MPI_Wtime() - comp_large_time;

//         // Stitches each bucket back together
//         for (i = 0; i < SIZE; i++) {
//             for (j = 0; j < buckets[i].size(); j++) {
//                 (*array)[index] = buckets[i][j];
//                 index++;
//             }
//         }


//         CALI_MARK_END(comp_large);
//         comp_large_time = MPI_Wtime() - comp_large_time;
//         CALI_MARK_END(comp);
//         comp_time = MPI_Wtime() - comp_time;

//         CALI_MARK_BEGIN(correctness_check);
//         correctness_check_time = MPI_Wtime();

//         sortedTest = correctnessCheck(&array);

//         CALI_MARK_END(correctness_check);
//         correctness_check_time = MPI_Wtime() - correctness_check_time;

//         if (sortedTest) {
//             printf("Array was correctly sorted.");
//         } else {
//             printf("Array was sorted incorrectly.");
//         }

//         printf("Array after being sorted:\n");
//         for (int i = 0; i < SIZE; i++) {
//             printf("array[%d] = %f", i, array[i]);
//         }


//     }


// /**************************** worker task ************************************/
//     if (taskid > MASTER)
//     {
//         vector<float> bucket;
//         bucket.push_back(0.0);
        
//         //RECEIVING PART FOR WORKER PROCESS STARTS HERE
//         CALI_MARK_BEGIN(worker_receive);
//         worker_receive_time = MPI_Wtime();

//         CALI_MARK_BEGIN(comm);
//         comm_time = MPI_Wtime() - comm_time;
//         CALI_MARK_BEGIN(comm_small);
//         comm_small_time = MPI_Wtime() - comm_small_time;

//         // Worker receives work from the master process
//         mtype = FROM_MASTER;
//         MPI_Recv(&(bucket[0]), 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        
//         CALI_MARK_END(comm_small);
//         comm_small_time = MPI_Wtime() - comm_small_time;
//         CALI_MARK_END(comm);
//         comm_time = MPI_Wtime() - comm_time;

//         //RECEIVING PART FOR WORKER PROCESS ENDS HERE
//         CALI_MARK_END(worker_receive);
//         worker_receive_time = MPI_Wtime() - worker_receive_time;

//         //CALCULATION PART FOR WORKER PROCESS STARTS HERE
//         CALI_MARK_BEGIN(worker_calculation);
//         worker_calculation_time = MPI_Wtime();

//         insertionSort(&(bucket[0]));
            
//         //CALCULATION PART FOR WORKER PROCESS ENDS HERE
//         CALI_MARK_END(worker_calculation);
//         worker_calculation_time = MPI_Wtime() - worker_calculation_time;
        
//         //SENDING PART FOR WORKER PROCESS STARTS HERE
//         CALI_MARK_BEGIN(worker_send);
//         worker_send_time = MPI_Wtime();

//         CALI_MARK_BEGIN(comm);
//         comm_time = MPI_Wtime() - comm_time;
//         CALI_MARK_BEGIN(comm_small);
//         comm_small_time = MPI_Wtime() - comm_small_time;

//         // Worker sends sorted data back to the master process
//         mtype = FROM_WORKER;
//         MPI_Send(&(bucket[0]), 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD)

//         CALI_MARK_END(comm_small);
//         comm_small_time = MPI_Wtime() - comm_small_time;
//         CALI_MARK_END(comm);
//         comm_time = MPI_Wtime() - comm_time;

//         //SENDING PART FOR WORKER PROCESS ENDS HERE
//         CALI_MARK_END(worker_send);
//         worker_send_time = MPI_Wtime() - worker_send_time;
//     }

//     // WHOLE PROGRAM COMPUTATION PART ENDS HERE
//     CALI_MARK_END(main);
//     main_time = MPI_Wtime() - main_time;

//     adiak::init(NULL);
//     adiak::user();
//     adiak::launchdate();
//     adiak::libraries();
//     adiak::cmdline();
//     adiak::clustername();
//     adiak::value("Algorithm", "BucketSort");
//     adiak::value("ProgrammingModel", "MPI"); 
//     adiak::value("Datatype", "float"); 
//     adiak::value("SizeOfDatatype", sizeof(float)); 
//     adiak::value("InputSize", SIZE); 
//     adiak::value("InputType", "Random"); 
//     adiak::value("num_procs", numtasks);
//     adiak::value("num_decimals", numDecimals);
//     adiak::value("program_name", "master_worker_bucket_sort");
//     // adiak::value("array_datatype_size", sizeof(float));
//     adiak::value("group_num", 17);
//     adiak::value("implementation_source", "Handwritten + Lab")


    

//     double worker_receive_time_max,
//         worker_receive_time_min,
//         worker_receive_time_sum,
//         worker_receive_time_average,
//         worker_calculation_time_max,
//         worker_calculation_time_min,
//         worker_calculation_time_sum,
//         worker_calculation_time_average,
//         worker_send_time_max,
//         worker_send_time_min,
//         worker_send_time_sum,
//         worker_send_time_average = 0; // Worker statistic values.

//     double comm_time_max,
//         comm_time_min,
//         comm_time_sum,
//         comm_time_average,
//         comm_small_time_max,
//         comm_small_time_min,
//         comm_small_time_sum,
//         comm_small_time_average,
//         comm_large_time_max,
//         comm_large_time_min,
//         comm_large_time_sum,
//         comm_large_time_average,
//         comp_time_max,
//         comp_time_min,
//         comp_time_sum,
//         comp_time_average,
//         comp_small_time_max,
//         comp_small_time_min,
//         comp_small_time_sum,
//         comp_small_time_average,
//         comp_large_time_max,
//         comp_large_time_min,
//         comp_large_time_sum,
//         comp_large_time_average = 0;
        

//     /* USE MPI_Reduce here to calculate the minimum, maximum and the average times for the worker processes.
//     MPI_Reduce (&sendbuf,&recvbuf,count,datatype,op,root,comm). https://hpc-tutorials.llnl.gov/mpi/collective_communication_routines/ */
    
//     // sendbuf = MPI_Wtime() counters
//     // recvbuf = new vars
//     // datatype -> MPI_DOUBLE
//     // op -> MPI_MAX, MPI_MIN, MPI_SUM
//     // There should be 9 total Reduce calls
    
//     MPI_Reduce(&worker_receive_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
//     MPI_Reduce(&worker_receive_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
//     MPI_Reduce(&worker_receive_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);
    
//     MPI_Reduce(&worker_calculation_time, &worker_calculation_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
//     MPI_Reduce(&worker_calculation_time, &worker_calculation_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
//     MPI_Reduce(&worker_calculation_time, &worker_calculation_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);
    
//     MPI_Reduce(&worker_send_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
//     MPI_Reduce(&worker_send_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
//     MPI_Reduce(&worker_send_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);

//     MPI_Reduce(&comm_time, &comm_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
//     MPI_Reduce(&comm_time, &comm_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
//     MPI_Reduce(&comm_time, &comm_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);

//     MPI_Reduce(&comm_small_time, &comm_small_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
//     MPI_Reduce(&comm_small_time, &comm_small_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
//     MPI_Reduce(&comm_small_time, &comm_small_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);

//     MPI_Reduce(&comp_time, &comp_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
//     MPI_Reduce(&comp_time, &comp_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
//     MPI_Reduce(&comp_time, &comp_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);

//     MPI_Reduce(&comp_small_time, &comp_small_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
//     MPI_Reduce(&comp_small_time, &comp_small_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
//     MPI_Reduce(&comp_small_time, &comp_small_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);


    
    
//     if (taskid == 0)
//     {
//         // Master Times
//         printf("******************************************************\n");
//         printf("Master Times:\n");
//         printf("Main Time: %f \n", main_time);
//         printf("Data Initialization Time: %f \n", data_init_time);
//         printf("Master Send and Receive Time: %f \n", master_send_receive_time);
//         printf("Master Correctness Check Time: %f \n", correctness_check_time)
//         printf("\n******************************************************\n");

//         // Add values to Adiak
//         adiak::value("MPI_Reduce-main_time", main_time);
//         adiak::value("MPI_Reduce-data_init_time", data_init_time);
//         adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);
//         adiak::value("MPI_Reduce-correctness_check_time", correctness_check_time);

//         // Must move values to master for adiak
//         mtype = FROM_WORKER;
//         MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&worker_receive_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&worker_send_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comm_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comm_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comm_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comm_small_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comm_small_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comm_small_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comp_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comp_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comp_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comp_small_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comp_small_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//         MPI_Recv(&comp_small_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

//         adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
//         adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
//         adiak::value("MPI_Reduce-worker_receive_time_average", worker_receive_time_average);
//         adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
//         adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
//         adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
//         adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
//         adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
//         adiak::value("MPI_Reduce-worker_send_time_average", worker_send_time_average);
//         adiak::value("MPI_Reduce-comm_time_max", comm_time_max);
//         adiak::value("MPI_Reduce-comm_time_min", comm_time_min);
//         adiak::value("MPI_Reduce-comm_time_average", comm_time_average);
//         adiak::value("MPI_Reduce-comm_small_time_max", comm_small_time_max);
//         adiak::value("MPI_Reduce-comm_small_time_min", comm_small_time_min);
//         adiak::value("MPI_Reduce-comm_small_time_average", comm_small_time_average);
//         adiak::value("MPI_Reduce-comp_time_max", comp_time_max);
//         adiak::value("MPI_Reduce-comp_time_min", comp_time_min);
//         adiak::value("MPI_Reduce-comp_time_average", comp_time_average);
//         adiak::value("MPI_Reduce-comp_small_time_max", comp_small_time_max);
//         adiak::value("MPI_Reduce-comp_small_time_min", comp_small_time_min);
//         adiak::value("MPI_Reduce-comp_small_time_average", comp_small_time_average);
        
//         adiak::value("sorted", sortedTest);
//     }
//     else if (taskid == 1)
//     { // Print only from the first worker.
//         // Print out worker time results.
        
//         // Compute averages after MPI_Reduce
//         worker_receive_time_average = worker_receive_time_sum / (double)numworkers;
//         worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
//         worker_send_time_average = worker_send_time_sum / (double)numworkers;

//         comm_time_average = comm_time_sum / (double)numworkers;
//         comm_small_time_average = comm_small_time_sum / (double)numworkers;
//         comp_time_average = comp_time_sum / (double)numworkers;
//         comp_small_time_average = comp_small_time_sum / (double)numworkers;

//         printf("******************************************************\n");
//         printf("Worker Times:\n");
//         printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
//         printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
//         printf("Worker Receive Time Average: %f \n", worker_receive_time_average);
//         printf("Worker Calculation Time Max: %f \n", worker_calculation_time_max);
//         printf("Worker Calculation Time Min: %f \n", worker_calculation_time_min);
//         printf("Worker Calculation Time Average: %f \n", worker_calculation_time_average);
//         printf("Worker Send Time Max: %f \n", worker_send_time_max);
//         printf("Worker Send Time Min: %f \n", worker_send_time_min);
//         printf("Worker Send Time Average: %f \n", worker_send_time_average);
//         printf("Comm Time Max: %f \n", comm_time_max);
//         printf("Comm Time Min: %f \n", comm_time_min);
//         printf("Comm Time Average: %f \n", comm_time_average);
//         printf("Comm Small Time Max: %f \n", comm_small_time_max);
//         printf("Comm Small Time Min: %f \n", comm_small_time_min);
//         printf("Comm Small Time Average: %f \n", comm_small_time_average);
//         printf("Comp Time Max: %f \n", comp_time_max);
//         printf("Comp Time Min: %f \n", comp_time_min);
//         printf("Comp Time Average: %f \n", comp_time_average);
//         printf("Comp Small Time Max: %f \n", comp_small_time_max);
//         printf("Comp Small Time Min: %f \n", comp_small_time_min);
//         printf("Comp Small Time Average: %f \n", comp_small_time_average);        
//         printf("\n******************************************************\n");

//         mtype = FROM_WORKER;
//         MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&worker_receive_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&worker_calculation_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&worker_send_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comm_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comm_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comm_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comm_small_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comm_small_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comm_small_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comp_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comp_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comp_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comp_small_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comp_small_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//         MPI_Send(&comp_small_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        
//     }

//     // Flush Caliper output before finalizing MPI
//     mgr.stop();
//     mgr.flush();

//     MPI_Finalize();
// }