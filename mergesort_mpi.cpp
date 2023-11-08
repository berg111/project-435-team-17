#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <malloc.h>
#include <mpi.h>


int merge(int*, int, int, int);
int mergeSort(int*, int, int);
int swap(int*, int, int);

int mergeSort(int *arr, int left, int right){
	if (left < right){
		int mid = (left + right) / 2;
		mergeSort(arr, left, mid);
		mergeSort(arr, mid + 1, right);
		merge(arr, left, mid, right);
	}
	return 0;
}

int merge(int *arr, int left, int mid, int right){
	int count = right - left + 1;
	int *tmp = (int*) calloc(count, sizeof(int));
	int i = left;
	int j = mid + 1;
	int k;

	for(k = 0; k < count; k++){
		if (i > mid){
			tmp[k] = arr[j];
			j++;
		}
		else if (j > right){
			tmp[k] = arr[i];
			i++;
		}
		else if (arr[i] < arr[j]){
			tmp[k] = arr[i];
			i++;
		}
		else {
			tmp[k] = arr[j];
			j++;
		}
	}

	for(k = 0; k < count; k++) {
		arr[left + k] = tmp[k];
	}
	
	free(tmp);
}

int swap(int *arr, int i, int j) {
	int tmp=arr[i];
	arr[i]=arr[j];
	arr[j]=tmp;
	return 0;
}

int main(int argc, char **argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	int mpirank = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int mpisize = MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status stat;

	int start = 0;
	int finish = 0;
	int n = 1000;
	int *arr;

	int range = n / size;

	if (rank == 0){
        arr = (int*) malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % n;
        }
	}

	int *arr_copy = (int*) calloc(range, sizeof(int));

	MPI_Scatter(arr, range, MPI_INT, arr_copy, range, MPI_INT, 0, MPI_COMM_WORLD);

	mergeSort(arr_copy, 0, range - 1);

	MPI_Gather(arr_copy, range, MPI_INT, arr, range, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {

		int parts = size;
		while (parts > 1) {
			for (int i = 0; i < parts; i += 2) {
				int left = i * (int) (n / parts);
				int mid = left + (int) (n / parts) - 1;
				int right = mid + (int) (n / parts);
				merge(arr, left, mid, right);
			}
			parts = parts >> 1;
		}
	}

	MPI_Finalize();

	bool passed = true;
    for(int i = 1; i < n; i++)
    {
        if (arr[i-1] > arr[i])
        {
            passed = false;
        }
    }
    printf( "Test %s\n", passed ? "PASSED" : "FAILED");

	if (rank == 0) {
		free(arr);
	}

	int int_size = sizeof(int);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", int_size); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", Random); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online + Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
	
    return 0;
}