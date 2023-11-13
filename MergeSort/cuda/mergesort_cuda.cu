#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

__global__ void MergeSort(int *nums, int *temp, int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 2; i < 2 * n; i *= 2) {
        int len = i;
        if (n - tid < len) len = n - tid;
        if (tid % i == 0) {
            int *seqA = &nums[tid], lenA = i / 2, j = 0;
            int *seqB = &nums[tid + lenA], lenB = len - lenA, k = 0;
            int p = tid;
            while (j < lenA && k < lenB) {
                if (seqA[j] < seqB[k]) {
                    temp[p++] = seqA[j++];
                } else {
                    temp[p++] = seqB[k++];
                }
            }
            while (j < lenA)
                temp[p++] = seqA[j++];
            while (k < lenB)
                temp[p++] = seqB[k++];
            for (int j = tid; j < tid + len; j++)
                nums[j] = temp[j];
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    // array_size = num_threads^2
    CALI_MARK_BEGIN("main");

    int array_size = atoi(argv[1]);
    std::string input_type = argv[2];
    int num_threads = atoi(argv[3]);

    // ************************ start data_init section ************************
    int *values = (int*)malloc(sizeof(int) * array_size);;

    if (input_type == "sorted") {
      for (int i = 0; i < array_size; i++) {
        values[i] = i;
      }
    } else if (input_type == "random") {
      for (int i = 0; i < array_size; i++) {
        values[i] = rand() % array_size;
      }
    } else if (input_type == "reverse") {
      for (int i = 0; i < array_size; i++) {
        values[i] = array_size - i;
      }
    } else if (input_type == "perturbed") {
      for (int i = 0; i < array_size; i++) {
        values[i] = i;
      }
      int num_perturbations = array_size / 100;
      for (int i = 0; i < num_perturbations; i++) {
        int rand_index1 = rand() % array_size;
        int rand_index2 = rand() % array_size;
        int temp = values[rand_index1];
        values[rand_index1] = values[rand_index2];
        values[rand_index2] = temp;
      }
    }
    // ************************ end data_init section ************************

    int *c_values;
    cudaMalloc((void**)&c_values, sizeof(int) * array_size);
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large_htd");
    cudaMemcpy(c_values, values, sizeof(int) * array_size, cudaMemcpyHostToDevice);
    CALI_MARK_END("comm_large_htd");
    CALI_MARK_END("comm");

    int *temp_values;
    cudaMalloc((void**)&temp_values, sizeof(int) * array_size);

    dim3 threadPerBlock(num_threads);
    dim3 blockNum((array_size + threadPerBlock.x - 1) / threadPerBlock.x);
    int num_blocks = (array_size + threadPerBlock.x - 1) / threadPerBlock.x;

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    MergeSort<<<blockNum, threadPerBlock>>>(c_values, temp_values, array_size);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large_dth");
    cudaMemcpy(values, c_values, sizeof(int) * array_size, cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large_dth");
    CALI_MARK_END("comm");

    // for (int i = 0; i < array_size; ++i) {
    //     printf("%d ", values[i]);
    // }
    printf("\n");


    // ************************ start correctness_check section ************************
    bool passed = true;
    for (int i = 1; i < array_size; i++) {
      if (values[i-1] > values[i]) {
        passed = false;
      }
    }
    printf("\nTest %s\n", passed ? "PASSED" : "FAILED");
    // ************************ end correctness_check section ************************

    free(values);
    cudaFree(c_values);
    cudaFree(temp_values);

    CALI_MARK_END("main");

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

}
