#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <vector>
#include <limits>

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
            printf("Random Array\n");

            randMax = pow(10, 5);
            for (int i = 0; i < size; i++) {
                array[i] = (float)((rand() % (randMax - 1)) + 1) / (float)randMax;
            }

            break;
        case 's':
            printf("Sorted Array\n");

            for (int i = 0; i < size; i++) {
                array[i] = (float)(i) / (float)(size + 1);
            }

            break;
        case 'R':
            printf("Reverse Sorted Array\n");

            counter = 0;
            for (int i = size-1; i >= 0; i--) {
                array[counter] = (float)(i) / (float)(size + 1);
                counter++;
            }

            break;
        case 'p':
            printf("1%% Perturbed Array\n");

            for (int i = 0; i < size; i++) {
                array[i] = (float)(i) / (float)(size + 1);
            }
            num_perturbations = size / 100;
            printf("num_perturbations = %d\n", num_perturbations);
            for (int i = 0; i < num_perturbations; i++) {
                    rand_index1 = rand() % size;
                rand_index2 = rand() % size;
                temp = array[rand_index1];
                array[rand_index1] = array[rand_index2];
                array[rand_index2] = temp;
            }


            break;
        default:
            printf("ERROR\n\n");
            break;
    }
}

__global__ void insertion_sort(float *device_data, int *device_start, int *device_offset){
    int start=device_start[blockIdx.x];
    int offset=device_offset[blockIdx.x];

    device_data += start;
    float tmp;
    int i, j, k;

    // Perform insertion sort
    for (i = 0; i < offset-1; i++) {
        j = i + 1;
        k = i;

        // find the smallest element
        for (j = i+1; j < offset; j++)
            if (device_data[k] > device_data[j])
                k = j;

        // swap elements
        tmp=device_data[k];
        device_data[k]=device_data[i];
        device_data[i]=tmp;
    }

    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // if (tid < 1024) {
    //     int start=device_start[tid];
    //     int offset=device_offset[tid];

    //     device_data += start;
    //     float tmp;
    //     int i, j, k;

    //     // Perform insertion sort
    //     for (i = 0; i < offset-1; i++) {
    //         j = i + 1;
    //         k = i;

    //         // find the smallest element
    //         for (j = i+1; j < offset; j++)
    //             if (device_data[k] > device_data[j])
    //                 k = j;

    //         // swap elements
    //         tmp=device_data[k];
    //         device_data[k]=device_data[i];
    //         device_data[i]=tmp;
    //     }
    // }
    
}

std::vector<vector <float> > get_buckets(int bucket_count) {
	std::vector<vector <float> > buckets;
    int i;
    for (i = 0;i < bucket_count; i++) {
        std::vector<float> list;
        buckets.push_back(list);
    }
    return buckets;
}

float get_max(int number_of_elements, float *data) {
	float max = -std::numeric_limits<float>::infinity();
    int i;
    for (i = 0; i < number_of_elements; i++)
        if(max < data[i])
            max = data[i];
    return max;
}

std::vector<vector <float> > assign_bucket(int number_of_elements, float *data, float max, int bucket_count) {
	std::vector<vector <float> > buckets = get_buckets(bucket_count);
	int index;
    int i;
	for (i = 0 ;i < number_of_elements; i++){
        index = int((bucket_count*data[i])/(max+0.01)); // same as used in pthreads
        buckets[index].push_back(data[i]);
    }
    return buckets;
}

void bucket_sort(std::vector<vector <float> > buckets, int bucket_count, 
					float *data, int *start, int *offset) {
	int size, i, j;
	int index = 0;
	for (i=0; i < bucket_count; i++) {
		size = buckets[i].size();
		offset[i]=int(buckets[i].size());
		start[i]=int(index);
		for (j=0; j < size; j++){
			data[index]=float(buckets[i][j]);
			index++;
		}
	}
}

int cuda_sort(int number_of_elements, float *a)
{
	int bucket_count;
	float *data, max;
	int *start, *offset;
	std::vector<vector <float> > buckets;

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
	// get bucket count - best performance found at this configuration
    // bucket_count = number_of_elements / 64;
    bucket_count = THREADS;
    // bucket_count=BLOCKS;

	// find max element
    max = get_max(number_of_elements, a);

    // assign elements to appropriate buckets 
    buckets = assign_bucket(number_of_elements, a, max, bucket_count);

    // perform bucket sort on the array by arranging the bucket elements
	data = (float *)malloc(number_of_elements*sizeof(float));
	start = (int *)malloc(bucket_count*sizeof(int));
	offset = (int *)malloc(bucket_count*sizeof(int));
	bucket_sort(buckets, bucket_count, data, start, offset);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

	// prepare for running insertion sorting on GPU in parallel
    float *device_data;
    int *device_start, *device_offset;

    cudaMalloc((void **) &device_data, sizeof(float)*number_of_elements);
    cudaMalloc((void **) &device_start, sizeof(int)*bucket_count);
    cudaMalloc((void **) &device_offset, sizeof(int)*bucket_count);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(device_data, data, sizeof(float)*number_of_elements, cudaMemcpyHostToDevice);
    cudaMemcpy(device_start, start, sizeof(int)*bucket_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_offset, offset, sizeof(int)*bucket_count, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // run sorting on GPU
    dim3 dimGrid(bucket_count, 1);
    dim3 dimBlock(1);
    // dim3 dimGrid(BLOCKS, 1);
    // dim3 dimBlock(THREADS, 1);
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    insertion_sort<<<dimGrid, dimBlock>>>(device_data, device_start, device_offset);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // copy results
    cudaMemcpy(a, device_data, sizeof(float)*number_of_elements, cudaMemcpyDeviceToHost);

    // free back to heap
    cudaFree(device_data);
    cudaFree(device_start);
    cudaFree(device_offset);
    free(data);
    free(start);
    free(offset);
	return 0;
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
    // BLOCKS = ceil((float)NUM_VALS / (float)THREADS);
    // BLOCKS = ceil((float)NUM_VALS/ (float)64);
    // NUM_BUCKETS = 1024;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n\n", NUM_VALS);
    // printf("Number of blocks: %d\n\n", BLOCKS);
    
    CALI_MARK_BEGIN(data_init);
    // Initialize array
    bytes = NUM_VALS * sizeof(float);
    float* array = (float*) malloc(bytes);
    char input = 'R';
    
    // Initialize data to be sorted
    initializeData(5, array, NUM_VALS, input);
    CALI_MARK_END(data_init);

    // Prints unsorted array
    // printf("Array BEFORE being sorted:\n");
    // printArray(array, NUM_VALS);

    // Call Bucket Sort
    cuda_sort(NUM_VALS, array);

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
    adiak::value("InputType", "ReverseSorted"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
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


