#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

int main(int argc, char** argv) {
	int n = atoi(argv[1]);
    int *arr = new int[n];

    int i;
    srand(time(NULL));
    printf("Unsorted array: ");
    for (i = 0; i < n; i++) {
        arr[i] = rand() % n;
        printf("%d ", arr[i]);
    }
    printf("\n");

    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int subarray_size = n / size;

    int *subarray = new int[subarray_size];
	// sending work out to all processes and subarrays
    MPI_Scatter(arr, subarray_size, MPI_INT, subarray, subarray_size, MPI_INT, 0, MPI_COMM_WORLD);

    int *temp_array = new int[subarray_size];
	// sorting the subarrays
    mergeSort(subarray, temp_array, 0, (subarray_size - 1));

    int *s_arr = NULL;
	// create new array for sorted result if main process
    if (rank == 0) {
        s_arr = new int[n];
    }

    MPI_Gather(subarray, subarray_size, MPI_INT, s_arr, subarray_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {

        int *finalArray = new int[n];
        mergeSort(s_arr, finalArray, 0, (n - 1));

        printf("Sorted array: ");
        for (i = 0; i < n; i++) {
            printf("%d ", finalArray[i]);
        }
        printf("\n");

        free(s_arr);
        free(finalArray);
    }

    free(arr);
    free(subarray);
    free(temp_array);

    MPI_Barrier(MPI_COMM_WORLD);
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
