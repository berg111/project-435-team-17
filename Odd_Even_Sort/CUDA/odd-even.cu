#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>

int NUM_VALUES;

__global__ void oddEvenSortStep(int* array, int size, int phase){


    for (size_t phase = 0; phase < size; phase++){

        //Even phase
        if(phase % 2 == 0){

            for(int i = 1; i < size - 1; i += 2){
                if (array[i] > array[i + 1]){
                    std::swap(array[i], array[i + 1]);

                }
            }
        }
        
        //Odd phase
        else{

            for(int i = 0; i < size - 1; i += 2){
                if (array[i] > array[i + 1]){
                    std::swap(array[i], array[i + 1]);

                }
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

    int threads = atoi(argv[0]);
    int arraySize = atoi(argv[1]);

    clock_t start, stop;

    std::vector<int> array;

    generate_array(array);

    int* gpu_array;
    size_t size = array.size() * sizeof(int);
    cudaMalloc((void**) &gpu_array, size);

    cudaMemcpy(gpu_array, array.data(), size, cudaMemcpyHostToDevice);

    start = clock();

    oddEvenSortKernel<<1, threads>>(gpu_array, arraySize);

    stop = clock();

    cudaMemcpy(array.data(), gpu_array, size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_array);



}