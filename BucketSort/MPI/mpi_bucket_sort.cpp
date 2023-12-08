#include <iostream>
#include <random>
#include <math.h>
#include <memory>
#include <numeric>
#include <algorithm>

#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int NUM_VALS;
int NUM_BUCKETS;

size_t bytes;

void initializeData(int numDecimalPlaces, float *array, int size, char input) 
{   
    long long int randMax;
    int counter;
    int num_perturbations;
    int rand_index1;
    int rand_index2;
    int temp;


    switch (input) {
        case 'r':
            cout << "Random Array" << endl;

            randMax = pow(10, 5);
            for (int i = 0; i < size; i++) {
                array[i] = (float)((rand() % (randMax - 1)) + 1) / (float)randMax;
            }

            break;
        case 's':
            cout << "Sorted Array" << endl;

            for (int i = 0; i < size; i++) {
                array[i] = (float)(i) / (float)(size + 1);
            }

            break;
        case 'R':
            cout << "Reverse Sorted Array" << endl;

            counter = 0;
            for (int i = size-1; i >= 0; i--) {
                array[counter] = (float)(i) / (float)(size + 1);
                counter++;
            }

            break;
        case 'p':
            cout << "1% Perturbed Array" << endl;

            for (int i = 0; i < size; i++) {
                array[i] = (float)(i) / (float)(size + 1);
            }
            num_perturbations = size / 100;
            cout << "num_perturbations = " << num_perturbations << endl;
            for (int i = 0; i < num_perturbations; i++) {
                    rand_index1 = rand() % size;
                rand_index2 = rand() % size;
                temp = array[rand_index1];
                array[rand_index1] = array[rand_index2];
                array[rand_index2] = temp;
            }


            break;
        default:
            cout << "ERROR\n" << endl;
            break;
    }
}

void insertionSort(float *array, int size, int offset) 
{
    int index;
    float temp;

    for (int i = 1; i < size; i++) {
        // Set index
        index = i;

        // Loop through elements behind current element
        while ((array[offset*size + index] < array[offset*size + (index - 1)]) && (array[offset*size + index] != -1.0)) {
            // Swap elements
            temp = array[offset*size + index];
            array[offset*size + index] = array[offset*size + (index - 1)];
            array[offset*size + (index - 1)] = temp;

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
    NUM_BUCKETS = 1024;

    // Initialize size of array
    bytes = NUM_VALS * sizeof(float);

    // Define Caliper region names
    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comm_small = "comm_small";
    const char* comm_large = "comm_large";
    const char* comp = "comp";
    const char* comp_small = "comp_small";
    const char* comp_large = "comp_large";
    const char* correctness_check = "correctness_check";

    // Rank timer
    double rank_time = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    rank_time = MPI_Wtime();

    // Get total number of tasks
    int num_tasks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    // Calculate chunk size
    // Assume this divides evenly
    const int chunk_size = NUM_BUCKETS * NUM_VALS / num_tasks;

    // Get the task ID
    int task_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    MPI_Comm newcomm;
    MPI_Comm_split(MPI_COMM_WORLD, task_id > 0, task_id - 1, &newcomm);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Create buffer for send only initialize in rank 0
    float* array;
    float* buckets;
    int index;

    // Generate random numbers from rank 0
    if (task_id == 0) {
        // Allocate memory for send buffer
        printf("Number of tasks: %d\n", num_tasks);
        printf("Number of values: %d\n", NUM_VALS);
        printf("Number of buckets: %d\n\n", NUM_BUCKETS);
        array = (float*) malloc(bytes);
        buckets = (float*) malloc(NUM_BUCKETS * bytes);

        char input = 'p';
        
        // Initialize Data
        CALI_MARK_BEGIN(data_init);
        initializeData(5, array,  NUM_VALS, input);
        CALI_MARK_END(data_init);

        // Initialize Buckets
        for (int i = 0; i < NUM_BUCKETS; i++) {
            for (int j = 0; j < NUM_VALS; j++) {
                buckets[i*NUM_VALS + j] = -1.0;
            }
        }

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        // Sort Buckets
        for (int i = 0; i < NUM_VALS; i++) {
            for (int j = 0; j < NUM_VALS; j++) {
                index = (int)(NUM_BUCKETS * array[i]);
                if (buckets[(index)*NUM_VALS + j] == -1.0) {
                    buckets[(index)*NUM_VALS + j] = array[i];
                    break;
                }
            }
        }
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        // printf("Buckets:\n");
        // for (int i = 0; i < NUM_BUCKETS; i++) {
        //     printf("[");
        //     for (int j = 0; j < NUM_VALS-1; j++) {
        //         if (buckets[i*NUM_VALS + j] != -1) {
        //             printf("%f, ", buckets[i*NUM_VALS + j]);
        //         }
        //     }
        //     printf("]\n");
        // }
        // printf("\n");
    }

    // Receive buffer
    float* recv_buffer = (float*) malloc(sizeof(float) * chunk_size);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    // Perform the scatter of the data to different threads
    MPI_Scatter(buckets, chunk_size, MPI_FLOAT, recv_buffer, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Calculate partial results in each thread
    int rank_buckets = chunk_size / NUM_VALS;

    if (task_id == 1) {
        cout << "# of elements per rank = " << chunk_size << endl;
    }
    
    // Sort each bucket
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    for (int i = 0; i < rank_buckets; i++) {
        insertionSort(recv_buffer, NUM_VALS, i);
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // Gather sorted buckets
    float* new_recv_ptr = (float*) malloc(sizeof(float) * NUM_VALS * NUM_BUCKETS);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather(recv_buffer, chunk_size, MPI_FLOAT, new_recv_ptr, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Print the result from rank 0
    if (task_id == 0) {
    
        // printf("Buckets:\n");
        // for (int i = 0; i < NUM_BUCKETS; i++) {
        //     printf("[");
        //     for (int j = 0; j < NUM_VALS-1; j++) {
        //         if (new_recv_ptr[i*NUM_VALS + j] != -1) {
        //             printf("%f, ", new_recv_ptr[i*NUM_VALS + j]);
        //         }
        //     }
        //     printf("]\n");
        // }
        // printf("\n");

        // Stitches each bucket back together
        index = 0;
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        for (int i = 0; i < NUM_BUCKETS; i++) {
            for (int j = 0; j < NUM_VALS; j++) {
                if (new_recv_ptr[i*NUM_VALS + j] == -1) {
                    break;
                }

                array[index] = new_recv_ptr[i*NUM_VALS + j];
                index++;
            }
        }
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        // Checks correctness of sorted array
        CALI_MARK_BEGIN(correctness_check);
        bool test = correctnessCheck(array, NUM_VALS);
        CALI_MARK_END(correctness_check);

        if (test) {
            printf("Array is correctly sorted!\n");
        } else {
            printf("Array is sorted incorrectly.\n\n");
            // printf("Array AFTER being sorted:\n");
            // printArray(array, NUM_VALS);
        } 

        free(buckets);
        free(array);

        cout << endl;

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
    }

    rank_time = MPI_Wtime() - rank_time;

    double rank_time_max,
      rank_time_min,
      rank_time_sum,
      rank_time_average;

    MPI_Reduce(&rank_time, &rank_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
    MPI_Reduce(&rank_time, &rank_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
    MPI_Reduce(&rank_time, &rank_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);

    if (task_id == 0) {

        MPI_Recv(&rank_time_max, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rank_time_min, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rank_time_average, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout << "Rank Time Max = " << rank_time_max << endl;
        cout << "Rank Time Min = " << rank_time_min << endl;
        cout << "Rank Time Avg = " << rank_time_average << endl;

        adiak::value("rank_time_max", rank_time_max);
        adiak::value("rank_time_min", rank_time_min);
        adiak::value("rank_time_average", rank_time_average);


    } else if (task_id == 1) {
        rank_time_average = rank_time_sum / (double)(num_tasks - 1);

        MPI_Send(&rank_time_max, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rank_time_min, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rank_time_average, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    }

    free(recv_buffer);
    free(new_recv_ptr);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}