#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

// size of list
#define NUM 64

__device__ inline
void merge(int* arr, int* new_arr, int left, int right, int upper_bound) {
  int i = left;
  int j = right;
  int k = left;

  while (i < right && j < upper_bound) { 
    if (arr[i] <= arr[j]) {
      new_arr[k] = arr[i];
      i++;
    } 
    else {
      new_arr[k] = arr[j];
      j++;
    }
    k++;
  }
  
  while (i < right) { 
    new_arr[k] = arr[i];
    i++;
    k++;
  }
  
  while (j < upper_bound) { 
    new_arr[k] = arr[j];
    j++;
    k++;
  }

  for (k = left; k < upper_bound; k++) { 
    arr[k] = new_arr[k]; 
  }
}

__global__ static void merge_sort(int* values, int* results) {
  
  extern __shared__ int shared[];
  const unsigned int tid = threadIdx.x;
  
  int k;
  int upper_bound;
  int i;

  shared[tid] = values[tid]; // input -> shared memory
  
  __syncthreads();
  
  k = 1;
  while(k < NUM) {
    i = 1;
    while(i + k <= NUM) {
        upper_bound = i + k * 2;
        if (upper_bound > NUM) {
          upper_bound = NUM + 1;
        }
        merge(shared, results, i, i + k, upper_bound);
        i = i + k * 2;
    }
    k = k * 2;
    __syncthreads();
  }
  
  values[tid] = shared[tid];
}

int main(int argc, char** argv) {
  
  int num_threads = NUM; //atoi(argv[1]);
  int num_blocks = 1; //BLOCKS = NUM_VALS / THREADS;

  cudaSetDevice(0); // Set the CUDA device you want to use

  int values[NUM];

  for (int i = 0; i < NUM; i++) {
    values[i] = rand() % NUM;
  }

  int* dvalues;
  int* results;
  cudaMalloc((void**)&dvalues, sizeof(int) * NUM);
  cudaMemcpy(dvalues, values, sizeof(int) * NUM, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&results, sizeof(int) * NUM);
  cudaMemcpy(results, values, sizeof(int)* NUM, cudaMemcpyHostToDevice);

  merge_sort<<<num_blocks, num_threads, sizeof(int) * num_threads*2>>>(dvalues, results); // num blocks, num threads per block, size of shared mem per block

  cudaDeviceSynchronize(); // Wait for GPU to finish

  cudaFree(dvalues);
  cudaMemcpy(values, results, sizeof(int) * NUM, cudaMemcpyDeviceToHost);
  cudaFree(results);

  bool passed = true;
  for(int i = 2; i < NUM; i++) {
    printf("\n%d", values[i-1]);
    if (values[i-1] > values[i]) {
      passed = false;
    }
  }
  printf("\nTest %s\n", passed ? "PASSED" : "FAILED");

  cudaDeviceReset(); // Reset the CUDA device

  int int_size = sizeof(int);

  // adiak::init(NULL);
  // adiak::launchdate();    // launch date of the job
  // adiak::libraries();     // Libraries used
  // adiak::cmdline();       // Command line used to launch the job
  // adiak::clustername();   // Name of the cluster
  // adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  // adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  // adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
  // adiak::value("SizeOfDatatype", int_size); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  // adiak::value("InputSize", NUM); // The number of elements in input dataset (1000)
  // adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
  // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
  // adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
  // adiak::value("implementation_source", "Online + Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

}
