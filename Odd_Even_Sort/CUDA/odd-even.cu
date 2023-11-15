#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>



int NUM_VALUES;
float timeA, timeB, timeC, bandwidth = 0.0;

__global__ void oddEvenSortStep(int* array, int size, int phase){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (phase % 2 == 0) {
        if (tid % 2 == 0 && tid < size - 1) {
            if (array[tid] > array[tid + 1]) {
                int temp = array[tid];
                array[tid] = array[tid + 1];
                array[tid + 1] = temp;
            }
        }
    } else {
        if (tid % 2 == 1 && tid < size - 1) {
            if (array[tid] > array[tid + 1]) {
                int temp = array[tid];
                array[tid] = array[tid + 1];
                array[tid + 1] = temp;
            }
        }
    }
}

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

void print_average(clock_t start, clock_t stop, int threads)
{
  double elapsed = (((double) (stop - start)) / CLOCKS_PER_SEC) / threads;
  printf("Average time: %.3fs\n", elapsed);
}




//Helper function to create array of random values
void generate_array(std::vector<int>& array, int size, int input){
    
    array.resize(size);

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

//Helper function to verify sort
bool isSorted(std::vector<int>& array){
    for(int i = 0; i < array.size() - 1; ++i){
        if(array[i] > array[i + 1]){
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv){

    CALI_CXX_MARK_FUNCTION;

    cudaEvent_t  sort_step_start, sort_step_end;

    int threads = atoi(argv[1]);
    int arraySize = atoi(argv[2]);
    int input = atoi(argv[3]);

    cali::ConfigManager mgr;
    mgr.start();

    dim3 blocks((arraySize + threads - 1) / threads, 1, 1);
    dim3 threadsPerBlock(threads, 1, 1);

    std::vector<int> array;

    generate_array(array, arraySize, input);
    std::cout << arraySize << "Threads: " << threads << "Input type: " << input << std::endl;

    // std::cout << "UNSorted Array: ";
    // for (int i = 0; i < array.size(); ++i) {
    //     std::cout << array[i] << " ";
    // }

    int* gpu_array;
    size_t size = array.size() * sizeof(int);
    cudaMalloc((void**) &gpu_array, size);

    cudaMemcpy(gpu_array, array.data(), size, cudaMemcpyHostToDevice);

    CALI_MARK_BEGIN("comp");
    cudaEventCreate(&sort_step_start);
    cudaEventCreate(&sort_step_end);
    cudaEventRecord(sort_step_start);

    clock_t start, stop;

    start = clock();

    for(int i = 0; i < array.size(); ++i){
        oddEvenSortStep<<<blocks, threadsPerBlock>>>(gpu_array, arraySize, i );
        cudaDeviceSynchronize();
    }

    stop = clock();

    print_elapsed(start, stop);
    print_average(start, stop, threads);

    cudaEventRecord(sort_step_end);
    cudaEventSynchronize(sort_step_end);
    cudaEventElapsedTime(&timeB, sort_step_start, sort_step_end);

    CALI_MARK_END("comp");

    cudaMemcpy(array.data(), gpu_array, size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_array);

    // printf("Time elapsed: %f ", timeB);

    if (isSorted(array)) {
        // std::cout << "Sorted Array: ";
        // for (int i = 0; i < array.size(); ++i) {
        //     std::cout << array[i] << " ";
        // }
        std::cout << "Array is sorted." << std::endl;
    } else {
        std::cout << "Array is not sorted." << std::endl;
        // std::cout << "Sorted Array: ";
        // for (int i = 0; i < array.size(); ++i) {
        //     std::cout << array[i] << " ";
        // }
        std::cout << std::endl;
    }

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_threads", threads);
    adiak::value("num_vals", arraySize);
    adiak::value("Sort_time", timeB);

    mgr.stop();
    mgr.flush();


}