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

size_t bytes;

const char* main = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";

void initializeData(int numDecimalPlaces, float *array, int size) 
{   
    long long int randMax = pow(10, numDecimalPlaces);

    for (int i = 0; i < size; i++) {
        array[i] = (float)((rand() % (randMax - 1)) + 1) / (float)randMax;
    }
}

__global__ void insertionSort(float *array, int size)
{
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Range check
    if (tid < size) {
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

void bucketSort(float *array, int size)
{
    // Sets index
    int index = 0;

    // Creates buckets
    float* buckets = (float*) malloc(bytes * NUM_VALS);

    float* dev_values;
    cudaMalloc((void**) &dev_values, bytes * size);

    // Filling buckets with null values
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            buckets[i*size + j] = -1.0;
        }
    }

    // Pushes array elements into buckets
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (buckets[((int)(size * array[i]))*size + j] == -1.0) {
                buckets[((int)(size * array[i]))*size + j] = array[i];
                break;
            }
        }
    }

    cout << "Buckets AFTER being filled:" << endl; 
    for (int i = 0; i < size; i++) {
        cout << "[";
        for (int j = 0; j < size-1; j++) {
            cout << buckets[i*size + j] << ", ";
        }
        cout << "]" << endl;
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(dev_values, buckets, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm_large);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    // Sorts each bucket with insertion sort
    insertionSort<<<BLOCKS, THREADS>>>(buckets, size);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // Sync with each thread before moving forward
    cudaDeviceSynchronize();

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(buckets, dev_values, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm_large);
    
    cudaFree(dev_values);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    // Stitches each bucket back together
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (buckets[i*size + j] == -1) {
                break;
            }

            array[index] = buckets[i*size + j];
            index++;
        }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
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


int main() 
{
    CALI_CXX_MARK_FUNCTION;

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Initialize threads, blocks, and number of values
    THREADS = 4;
    NUM_VALS = 4;
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n\n", BLOCKS);

    CALI_MARK_BEGIN(main);
    
    CALI_MARK_BEGIN(data_init);
    // Initialize array
    bytes = NUM_VALS * sizeof(bytes);
    float* array = (float*) malloc(bytes);
    
    // Initialize data to be sorted
    initializeData(5, array, NUM_VALS);
    CALI_MARK_END(data_init);

    // Prints unsorted array
    // printf("Array BEFORE being sorted:\n");
    // for (int i = 0; i < NUM_VALS; i++) {
    //     printf("array[%d] = %f\n", i, array[i]);
    // }
    // cout << endl;

    // Call Bucket Sort
    bucketSort(array, NUM_VALS);

    CALI_MARK_BEGIN(correctness_check);
    // Checks correctness of sorted array
    bool test = correctnessCheck(array, NUM_VALS);
    CALI_MARK_END(correctness_check);

    CALI_MARK_END(main);

    if (test) {
        printf("Array is correctly sorted.\n");
    } else {
        printf("Array is sorted incorrectly.\n\n");
        printf("Array AFTER being sorted:\n");
        for (int i = 0; i < NUM_VALS; i++) {
            printf("array[%d] = %f\n", i, array[i]);
        }
    }

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
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", "17"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online & Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").



    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    return 0;
}