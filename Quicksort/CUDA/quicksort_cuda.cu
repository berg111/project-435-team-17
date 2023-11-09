#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int NUM_VALUES;

__device__ void swap(std::vector<int>& array, int l, int r) {
    int temp = array[a];
    array[a] = array[b];
    array[b] = temp;
}

__global__ void quicksort(std::vector<int>& array, int l, int r){
    if (l < r) {
        int pivot = array[r];
        int index = l - 1;

        for (int i = l; i < r; i++) {
            if (array[i] <= pivot) {
                index++;
                swap(array[index], array[i]);
            }
        }

        std::swap(array[index + 1], array[r]);
        int part = index + 1;

        quicksort<<1, 1>>(array, l, part - 1);
        quicksort<<1, 1>>(array, part + 1, r);
        
    }
}

//Helper function to create array of random values
void generate_array(std::vector<int>& array, int size){
    
    array.resize(size);

	for(int i = 0; i < size; i++){
		array[i] = rand() % 100000;
    }
}

bool isSorted(std::vector<int>& array){
    for(int i = 0; i < array.size() - 1; ++i){
        if(array[i] > array[i + 1]){
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv){

    int threads = atoi(argv[0]);
    int array_size = atoi(argv[1]);

    cali::ConfigManager mgr;
    mgr.start();

    clock_t start, stop;

    std::vector<int> array;

    generate_array(array);

    int* gpu_array;
    size_t size = array.size() * sizeof(int);
    cudaMalloc((void**) &gpu_array, size);

    cudaMemcpy(gpu_array, array.data(), size, cudaMemcpyHostToDevice);

    start = clock();

    quicksort<<1, threads>>(gpu_array, array_size, i);

    stop = clock();

    double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;




    cudaMemcpy(array.data(), gpu_array, size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_array);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_threads", threads);
    adiak::value("num_vals", array_size);
    adiak::value("Sort_time", elapsed);

    mgr.stop();
    mgr.flush();


}