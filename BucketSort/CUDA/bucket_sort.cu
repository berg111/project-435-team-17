#include <cstdlib>
#include <cassert>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;
int NUM_BUCKETS;

size_t bytes;

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";

void initializeData(int numDecimalPlaces, float *array, long int size, char input) 
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

__global__ void insertionSort(float *array, long int size, int NUM_BUCKETS)
{
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Range check
    if (tid < NUM_BUCKETS) {
        // Do work

        int index;
        float temp;

        for (int i = 1; i < size; i++) {
            // Set index
            index = i;

            // Loop through elements behind current element
            while ((array[tid*size + index] < array[tid*size + (index - 1)]) && (array[tid*size + index] != -1.0)) {
                // Swap elements
                temp = array[tid*size + index];
                array[tid*size + index] = array[tid*size + (index - 1)];
                array[tid*size + (index - 1)] = temp;

                // Change index
                index--;
                
                // End condition
                if (index == 0) {
                    break;
                }
            }
        }
    }
}

void printBuckets(float* buckets, long int size) 
{
    printf("Buckets:\n");
    for (int i = 0; i < NUM_BUCKETS; i++) {
        printf("[");
        for (int j = 0; j < size-1; j++) {
            if (buckets[i*size + j] != -1) {
                printf("%f, ", buckets[i*size + j]);
            }
        }
        printf("]\n");
    }
    printf("\n");
}

void bucketSort(float *array, long int size)
{
    // Sets index
    int index = 0;

    // Creates buckets
    long int SIZE = size/2;
    const long int BYTES = SIZE * sizeof(float) * NUM_BUCKETS;
    cout << "Allocating " << BYTES << " bytes for buckets" << endl;
    float* buckets = new float[BYTES];
    const int temp = sizeof(int) * NUM_BUCKETS;
    int* counts = new int[temp];

    float* dev_values;
    cudaMalloc((void**) &dev_values, BYTES);

    cout << "Filling buckets..." << endl;
    // Filling buckets with null values
    for (int i = 0; i < NUM_BUCKETS; i++) {
        for (int j = 0; j < SIZE; j++) {
            buckets[i*SIZE + j] = -1.0;
        }
    }

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    // Pushes array elements into buckets
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < SIZE; j++) {
            index = (int)(NUM_BUCKETS * array[i]);
            if (buckets[(index)*SIZE + j] == -1.0) {
                //cout << "Sorting " << array[i] << " into bucket " << index << " at place " << j << endl;
                buckets[(index)*SIZE + j] = array[i];
                counts[index]++;
                break;
            }
        }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    //printBuckets(buckets, size);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    // Copy contents of buckets over to device array dev_values
    cudaMemcpy(dev_values, buckets, BYTES, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);


    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */


    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    // Sorts each bucket with insertion sort
    insertionSort<<<blocks, threads>>>(dev_values, SIZE, NUM_BUCKETS);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);


    // Sync with each thread before moving forward
    cudaDeviceSynchronize();


    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    // Copy contents of device array dev_values back to the host buckets
    cudaMemcpy(buckets, dev_values, BYTES, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);


    cudaFree(dev_values);
    cudaDeviceSynchronize();

    //printBuckets(buckets, size);

    index = 0;
    
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    // Stitches each bucket back together
    for (int i = 0; i < NUM_BUCKETS; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (buckets[i*SIZE + j] == -1) {
                break;
            }

            array[index] = buckets[i*SIZE + j];
            index++;
        }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);


    free(buckets);
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


int main(int argc, char *argv[]) 
{
    CALI_CXX_MARK_FUNCTION;

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Initialize threads, blocks, and number of values
    THREADS = atoi(argv[1]); // Max value is 1024, DO NOT GO OVER
    NUM_VALS = atoi(argv[2]); // Max value is 16383 or 2^14 - 1, DO NOT GO OVER
    BLOCKS = ceil((float)NUM_VALS / (float)THREADS);
    NUM_BUCKETS = 1024;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n\n", BLOCKS);
    
    CALI_MARK_BEGIN(data_init);
    // Initialize array
    bytes = NUM_VALS * sizeof(float);
    float* array = (float*) malloc(bytes);
    char input = 'p';
    
    // Initialize data to be sorted
    initializeData(5, array, NUM_VALS, input);
    CALI_MARK_END(data_init);

    // Prints unsorted array
    // printf("Array BEFORE being sorted:\n");
    // printArray(array, NUM_VALS);

    // Call Bucket Sort
    bucketSort(array, NUM_VALS);

    CALI_MARK_BEGIN(correctness_check);
    // Checks correctness of sorted array
    bool test = correctnessCheck(array, NUM_VALS);
    CALI_MARK_END(correctness_check);


    if (test) {
        printf("Array is correctly sorted!\n");
    } else {
        printf("Array is sorted incorrectly.\n\n");
        // printf("Array AFTER being sorted:\n");
        // printArray(array, NUM_VALS);
    }

    //printArray(array, NUM_VALS);

    free(array);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Bucket Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "1%%perturbed"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", "17"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online & Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").



    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    return 0;
}