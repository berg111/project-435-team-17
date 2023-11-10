#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string>

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

    int array_size = atoi(argv[1]);
    std::string input_type = argv[2];
    int num_threads = atoi(argv[3]);

    // Create the input
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

    int *c_values;
    cudaMalloc((void**)&c_values, sizeof(int) * array_size);
    cudaMemcpy(c_values, values, sizeof(int) * array_size, cudaMemcpyHostToDevice);

    int *temp_values;
    cudaMalloc((void**)&temp_values, sizeof(int) * array_size);

    dim3 threadPerBlock(num_threads);
    dim3 blockNum((array_size + threadPerBlock.x - 1) / threadPerBlock.x);
    MergeSort<<<blockNum, threadPerBlock>>>(c_values, temp_values, array_size);

    cudaMemcpy(values, c_values, sizeof(int) * array_size, cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < array_size; ++i) {
        printf("%d ", values[i]);
    }
    printf("\n");

    bool passed = true;
    for (int i = 1; i < array_size; i++) {
      if (values[i-1] > values[i]) {
        passed = false;
      }
    }
    printf("\nTest %s\n", passed ? "PASSED" : "FAILED");

    free(values);
    cudaFree(c_values);
    cudaFree(temp_values);

}
