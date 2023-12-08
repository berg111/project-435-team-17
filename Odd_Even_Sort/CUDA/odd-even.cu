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

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";

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

// void print_average(clock_t start, clock_t stop, int threads)
// {
//   double elapsed = (((double) (stop - start)) / CLOCKS_PER_SEC) / threads;
//   printf("Average time: %.3fs\n", elapsed);
// }




//Helper function to create array of random values
void generate_array(std::vector<int>& array, int size, int input){
    
    array.resize(size);
    std::vector<int> temp;
    temp.resize(size);
	switch(input){
        
        int aaa;
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

            for (int i = 0; i < size; i++) {
                temp[i] = i;
            }
            for(int i = 0; i < size; ++i){
                array[i] = temp[size - 1 - i];
            }
            break;
        case 4:

            for (int i = 0; i < size; i++) {
                array[i] = i;
            }
            int num_perturbations = size / 100;
            for (int i = 0; i < num_perturbations; i++) {
                int rand_index1 = rand() % size;
                int rand_index2 = rand() % size;
                aaa = array[rand_index1];
                array[rand_index1] = array[rand_index2];
                array[rand_index2] = aaa;
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

    int threadss = atoi(argv[1]);
    int arraySize = atoi(argv[2]);
    int input = atoi(argv[3]);
    int blockss = arraySize / threadss;
    std::string inputType;
    if(input == 1){
        inputType = "Random";
    }
    else if(input == 2){
        inputType = "Sorted";
    }
    else if(input == 3){
        inputType = "Reverse Sorted";
    }
    else if(input == 4){
        inputType = "1%perturbed";
    }
    cali::ConfigManager mgr;
    mgr.start();

    dim3 blocks(blockss, 1);
    dim3 threads(threadss, 1);

    std::vector<int> array;
    CALI_MARK_BEGIN(data_init);
    generate_array(array, arraySize, input);
    
    std::cout << arraySize << "Threads: " << threadss << "Input type: " << input << std::endl;

    // std::cout << "UNSorted Array: ";
    // for (int i = 0; i < array.size(); ++i) {
    //     std::cout << array[i] << " ";
    // }

    int* gpu_array;
    size_t size = array.size() * sizeof(int);
    cudaMalloc((void**) &gpu_array, size);
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(gpu_array, array.data(), size, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    cudaEventCreate(&sort_step_start);
    cudaEventCreate(&sort_step_end);
    cudaEventRecord(sort_step_start);

    clock_t start, stop;

    start = clock();

    for(int i = 0; i < array.size(); ++i){
        oddEvenSortStep<<<blocks, threads>>>(gpu_array, arraySize, i );
        cudaDeviceSynchronize();
    }

    stop = clock();

    // print_elapsed(start, stop);
    // print_average(start, stop, threads);

    cudaEventRecord(sort_step_end);
    cudaEventSynchronize(sort_step_end);
    cudaEventElapsedTime(&timeB, sort_step_start, sort_step_end);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(array.data(), gpu_array, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    cudaFree(gpu_array);

    // printf("Time elapsed: %f ", timeB);
    CALI_MARK_BEGIN(correctness_check);
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
    CALI_MARK_END(correctness_check);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Odd-Even Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", arraySize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", threadss); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", blockss); // The number of CUDA blocks 
    adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online/AI/Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();


}