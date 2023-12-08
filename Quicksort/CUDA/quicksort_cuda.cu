#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>



#define BLOCK_SIZE 256
float timeA, timeB, timeC, bandwidth = 0.0;

// Function to swap two elements
__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Function to perform partition
__device__ int partition(int *arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// Function for iterative Quicksort using a stack
__global__ void quicksort(int *arr, int low, int high) {
    int stack[32]; // Arbitrary size for the stack
    int top = -1;

    stack[++top] = low;
    stack[++top] = high;

    while (top >= 0) {
        high = stack[top--];
        low = stack[top--];

        int pi = partition(arr, low, high);

        if (pi - 1 > low) {
            stack[++top] = low;
            stack[++top] = pi - 1;
        }

        if (pi + 1 < high) {
            stack[++top] = pi + 1;
            stack[++top] = high;
        }
    }
}




//Helper function to create array of random values
void generate_array(int* array, int size){
    
    // array.resize(size);

	for(int i = 0; i < size; i++){
		array[i] = i;
    }
}

bool isSorted(int* array, int array_size){
    for(int i = 0; i < array_size - 1; ++i){
        if(array[i] > array[i + 1]){
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv){

    CALI_CXX_MARK_FUNCTION;

    cudaEvent_t h_t_d_start, h_t_d_end, sort_step_start, sort_step_end, d_t_h_start, d_t_h_end;

    int threads = atoi(argv[1]);
    int array_size = atoi(argv[2]);

    cali::ConfigManager mgr;
    mgr.start();

    // clock_t start, stop;

    int array[array_size] = {0};

    generate_array(array, array_size);
    // for (int i = 0; i < array_size; i++) {
    //     std::cout << array[i] << ", ";
    // }
    std::cout << std::endl;
    std::cout << array_size << std::endl;

    int* gpu_array;
    size_t size = array_size * sizeof(int);
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**) &gpu_array, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }
    cudaStatus = cudaMemcpy(gpu_array, array, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy HostToDevice failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(gpu_array);
        return 1;
    }
    // start = clock();

    CALI_MARK_BEGIN("comp");
    cudaEventCreate(&sort_step_start);
    cudaEventCreate(&sort_step_end);
    cudaEventRecord(sort_step_start);

    quicksort<<<1, threads>>>(gpu_array, 0, array_size - 1);
    cudaDeviceSynchronize();


    cudaEventRecord(sort_step_end);
    cudaEventSynchronize(sort_step_end);
    cudaEventElapsedTime(&timeB, sort_step_start, sort_step_end);

    CALI_MARK_END("comp");
    // stop = clock();

    // double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;




    cudaStatus = cudaMemcpy(array, gpu_array, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy DeviceToHost failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(gpu_array);
        return 1;
    }

    printf("Time elapsed: %f ", timeB / 1000);

    if (isSorted(array, array_size)) {
        std::cout << "Array is sorted." << std::endl;
    } else {
        std::cout << "Array is not sorted." << std::endl;
    }
    // for (int i = 0; i < array_size; i++) {
    //     std::cout << array[i] << ", ";
    // }

    cudaFree(gpu_array);

    // adiak::init(NULL);
    // adiak::user();
    // adiak::launchdate();
    // adiak::libraries();
    // adiak::cmdline();
    // adiak::clustername();
    // adiak::value("num_threads", threads);
    // adiak::value("num_vals", array_size);
    // // adiak::value("Sort_time", elapsed);

            adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "QuickSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
        adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_threads", threads); // The number of processors (MPI ranks)
        // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
        // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
        adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();


}