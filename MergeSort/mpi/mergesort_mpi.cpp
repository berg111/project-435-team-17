#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
	int n = atoi(argv[1]);
	std::string input_type = argv[2];
    int *arr = new int[n];

    srand(time(NULL));

	// ************************ start data_init section **********************
	CALI_MARK_BEGIN("data_init");
	if (input_type == "sorted") {
		for (int i = 0; i < n; i++) {
			arr[i] = i;
		}
	} else if (input_type == "random") {
		for (int i = 0; i < n; i++) {
			arr[i] = rand() % n;
		}
	} else if (input_type == "reverse") {
		for (int i = 0; i < n; i++) {
			arr[i] = n - i;
		}
	} else if (input_type == "perturbed") {
		for (int i = 0; i < n; i++) {
			arr[i] = i;
		}
		int num_perturbations = n / 100;
		for (int i = 0; i < num_perturbations; i++) {
			int rand_index1 = rand() % n;
			int rand_index2 = rand() % n;
			int temp = arr[rand_index1];
			arr[rand_index1] = arr[rand_index2];
			arr[rand_index2] = temp;
		}
	}
    CALI_MARK_END("data_init");
    // ************************ end data_init section ************************

    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	// if (rank == 0) {
	// 	printf("Original array: ");
	// 	for (int i = 0; i < n; i++) {
	// 		printf("%d ", arr[i]);
	// 	}
	// 	printf("\n");
	// }

    int subarray_size = n / size;

    int *subarray = new int[subarray_size];

	// sending work out to all processes and subarrays
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large_scatter");
    MPI_Scatter(arr, subarray_size, MPI_INT, subarray, subarray_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large_scatter");
    CALI_MARK_END("comm");

    int *temp_array = new int[subarray_size];

	// sorting the subarrays
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    mergeSort(subarray, temp_array, 0, (subarray_size - 1));
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    int *s_arr = NULL;
	// create new array for sorted result if main process
    if (rank == 0) {
        s_arr = new int[n];
    }

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large_gather");
    MPI_Gather(subarray, subarray_size, MPI_INT, s_arr, subarray_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large_gather");
    CALI_MARK_END("comm");

    if (rank == 0) {

        int *finalArray = new int[n];
        mergeSort(s_arr, finalArray, 0, (n - 1));

        // printf("Sorted array: ");
        // for (int i = 0; i < n; i++) {
        //     printf("%d ", finalArray[i]);
        // }
        printf("\n");

        // ************************ start correctness_check section ************************
        bool passed = true;
        for (int i = 1; i < n; i++) {
            if (finalArray[i-1] > finalArray[i]) {
                passed = false;
            }
        }
        printf("\nTest %s\n", passed ? "PASSED" : "FAILED");
        // ************************ end correctness_check section ************************

        free(s_arr);
        free(finalArray);

        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", n); // The number of elements in input dataset (1000)
        adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", size); // The number of processors (MPI ranks)
        adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    }

    free(arr);
    free(subarray);
    free(temp_array);

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END("comm_barrier");
    CALI_MARK_END("comm");

    MPI_Finalize();
}

void merge(int *left_array, int *right_array, int left, int middle, int right) {

    int h, i, j, k;
    h = left;
    i = left;
    j = middle + 1;

    while ((h <= middle) && (j <= right)) {
        if (left_array[h] <= left_array[j]) {
            right_array[i] = left_array[h];
            h++;
        }
        else {
            right_array[i] = left_array[j];
            j++;
        }
        i++;
    }

    if (middle < h) {
        for (k = j; k <= right; k++) {
            right_array[i] = left_array[k];
            i++;
        }
    }
    else {
        for (k = h; k <= middle; k++) {
            right_array[i] = left_array[k];
            i++;
        }
    }

    for (k = left; k <= right; k++) {
        left_array[k] = right_array[k];
    }
}

void mergeSort(int *left_array, int *right_array, int left, int right) {
    int middle;
    if (left < right) {
        middle = (left + right) / 2;
        mergeSort(left_array, right_array, left, middle);
        mergeSort(left_array, right_array, middle + 1, right);
        merge(left_array, right_array, left, middle, right);
    }
}
