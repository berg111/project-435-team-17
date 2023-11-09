#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>

// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>



int NUM_VALUES;

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


//Helper function to create array of random values
void generate_array(std::vector<int>& array, int size){
    
    array.resize(size);

	for(int i = 0; i < size; i++){
		array[i] = rand() % 100000;
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

    int threads = atoi(argv[1]);
    int arraySize = atoi(argv[2]);

    // cali::ConfigManager mgr;
    // mgr.start();

    dim3 blocks((arraySize + threads - 1) / threads, 1, 1);
    dim3 threadsPerBlock(threads, 1, 1);

    clock_t start, stop;

    std::vector<int> array;

    generate_array(array, arraySize);
    std::cout << arraySize << std::endl;
    std::cout << "UNSorted Array: ";
    // for (int i = 0; i < array.size(); ++i) {
    //     std::cout << array[i] << " ";
    // }

    int* gpu_array;
    size_t size = array.size() * sizeof(int);
    cudaMalloc((void**) &gpu_array, size);

    cudaMemcpy(gpu_array, array.data(), size, cudaMemcpyHostToDevice);

    start = clock();

    for(int i = 0; i < array.size(); ++i){
        oddEvenSortStep<<<blocks, threadsPerBlock>>>(gpu_array, arraySize, i );
        cudaDeviceSynchronize();
    }

    stop = clock();

    double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;




    cudaMemcpy(array.data(), gpu_array, size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_array);




    if (isSorted(array)) {
        std::cout << "Sorted Array: ";
        for (int i = 0; i < array.size(); ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << "Array is sorted." << std::endl;
    } else {
        std::cout << "Array is not sorted." << std::endl;
        std::cout << "Sorted Array: ";
        for (int i = 0; i < array.size(); ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }

    // adiak::init(NULL);
    // adiak::user();
    // adiak::launchdate();
    // adiak::libraries();
    // adiak::cmdline();
    // adiak::clustername();
    // adiak::value("num_threads", threads);
    // adiak::value("num_vals", arraySize);
    // adiak::value("Sort_time", elapsed);

    // mgr.stop();
    // mgr.flush();


}