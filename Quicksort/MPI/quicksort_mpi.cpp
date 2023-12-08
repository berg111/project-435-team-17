#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0           /* taskid of the first task */
#define FROM_MASTER 1      /* setting a message type for sending data to workers */
#define FROM_WORKER 2      /* setting a message type for receiving data from master */


void generate_array(int array[], int size, int input) {
    long long int randMax;
    int counter;
    int num_perturbations;
    int rand_index1;
    int rand_index2;
    int temp;
    switch(input){
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
        {
            counter = 0;
            for (int i = size-1; i >= 0; i--) {
                array[counter] = (float)(i) / (float)(size + 1);
                counter++;
            }
            break;
        }
        case 4:
            {
            for (int i = 0; i < size; i++) {
                array[i] = (float)(i) / (float)(size + 1);
            }
            num_perturbations = size / 100;
            for (int i = 0; i < num_perturbations; i++) {
                rand_index1 = rand() % size;
                rand_index2 = rand() % size;
                temp = array[rand_index1];
                array[rand_index1] = array[rand_index2];
                array[rand_index2] = temp;
            }
            }


            break;
    }

}


void swap(int* arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

// Function that performs the Quick Sort
// for an array arr[] starting from the
// index start and ending at index end
void quicksort(int* arr, int start, int end) {
    int pivot, index;

    // Base Case
    if (end <= 1)
        return;

    // Pick pivot and swap with first
    // element Pivot is middle element
    pivot = arr[start + end / 2];
    swap(arr, start, start + end / 2);

    // Partitioning Steps
    index = start;

    // Iterate over the range [start, end]
    for (int i = start + 1; i < start + end; i++) {

        // Swap if the element is less
        // than the pivot element
        if (arr[i] < pivot) {
            index++;
            swap(arr, i, index);
        }
    }

    // Swap the pivot into place
    swap(arr, start, index);

    // Recursive Call for sorting
    // of quick sort function
    quicksort(arr, start, index - start);
    quicksort(arr, index + 1, start + end - index - 1);
}

// Function that merges the two arrays
int* merge(int* arr1, int n1, int* arr2, int n2) {
    int* result = new int[n1 + n2];
    int i = 0;
    int j = 0;
    int k;

    for (k = 0; k < n1 + n2; k++) {
        if (i >= n1) {
            result[k] = arr2[j];
            j++;
        }
        else if (j >= n2) {
            result[k] = arr1[i];
            i++;
        }
        else if (arr1[i] < arr2[j]) {
            result[k] = arr1[i];
            i++;
        }
        else {
            result[k] = arr2[j];
            j++;
        }
    }
    return result;
}

bool isSorted(int* array, int array_size){
    for(int i = 0; i < array_size - 1; ++i){
        if(array[i] > array[i + 1]){
            return false;
        }
    }
    return true;
}
 
// Driver Code
int main(int argc, char* argv[])
{

    CALI_CXX_MARK_FUNCTION;

    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comm_small = "comm_small";
    const char* comm_large = "comm_large";
    const char* comp = "comp";
    const char* comp_small = "comp_small";
    const char* comp_large = "comp_large";
    const char* correctness_check = "correctness_check";

    // int number_of_elements;
    int chunk_size, own_chunk_size;
    int* chunk;
    FILE* file = NULL;
    double time_taken;
    MPI_Status status;

    int number_of_elements = std::atoi(argv[1]);
    // printf("%d", number_of_elements);
    int* data = new int[number_of_elements];


    // generate_array(data, number_of_elements, 1);
 
    int number_of_process, rank_of_process;
    int rc = MPI_Init(&argc, &argv);
 
    if (rc != MPI_SUCCESS) {
        printf("Error in creating MPI "
               "program.\n "
               "Terminating......\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
 
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);
 
    if (rank_of_process == 0) {
        CALI_MARK_BEGIN(data_init);
        generate_array(data, number_of_elements, 4);
        CALI_MARK_END(data_init);
        // Computing chunk size
        chunk_size
            = (number_of_elements % number_of_process == 0)
                  ? (number_of_elements / number_of_process)
                  : (number_of_elements / number_of_process
                     - 1);
 
        data = (int*)malloc(number_of_process * chunk_size
                            * sizeof(int));
 

 
        // Padding data with zero
        for (int i = number_of_elements;
             i < number_of_process * chunk_size; i++) {
            data[i] = 0;
        }
 
        printf("\n");
    }
 
    // Blocks all process until reach this point
    MPI_Barrier(MPI_COMM_WORLD);
 
    // Starts Timer
    time_taken -= MPI_Wtime();
 
    // BroadCast the Size to all the
    // process from root process
    // MPI_Bcast(&number_of_elements, 1, MPI_INT, 0,
    //           MPI_COMM_WORLD);
 
    // Computing chunk size
    chunk_size
        = (number_of_elements % number_of_process == 0)
              ? (number_of_elements / number_of_process)
              : number_of_elements
                    / (number_of_process - 1);
 
    // Calculating total size of chunk
    // according to bits
    chunk = (int*)malloc(chunk_size * sizeof(int));
 
    // Scatter the chuck size data to all process
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Scatter(data, chunk_size, MPI_INT, chunk,
                chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    free(data);
    data = NULL;
 
    // Compute size of own chunk and
    // then sort them
    // using quick sort
 
    own_chunk_size = (number_of_elements
                      >= chunk_size * (rank_of_process + 1))
                         ? chunk_size
                         : (number_of_elements
                            - chunk_size * rank_of_process);
 
    // Sorting array with quick sort for every
    // chunk as called by process
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    quicksort(chunk, 0, own_chunk_size);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
 
    for (int step = 1; step < number_of_process;
         step = 2 * step) {
        if (rank_of_process % (2 * step) != 0) {
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_small);
            MPI_Send(chunk, own_chunk_size, MPI_INT,
                     rank_of_process - step, 0,
                     MPI_COMM_WORLD);
            CALI_MARK_END(comm_small);
            CALI_MARK_END(comm);
            
            break;
        }
 
        if (rank_of_process + step < number_of_process) {
            int received_chunk_size
                = (number_of_elements
                   >= chunk_size
                          * (rank_of_process + 2 * step))
                      ? (chunk_size * step)
                      : (number_of_elements
                         - chunk_size
                               * (rank_of_process + step));
            int* chunk_received;
            chunk_received = (int*)malloc(
                received_chunk_size * sizeof(int));
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_small);
            MPI_Recv(chunk_received, received_chunk_size,
                     MPI_INT, rank_of_process + step, 0,
                     MPI_COMM_WORLD, &status);
            CALI_MARK_END(comm_small);
            CALI_MARK_END(comm);

            CALI_MARK_BEGIN(comp);
            CALI_MARK_BEGIN(comp_large);
            data = merge(chunk, own_chunk_size,
                         chunk_received,
                         received_chunk_size);
            CALI_MARK_END(comp_large);
            CALI_MARK_END(comp);
 
            free(chunk);
            free(chunk_received);
            chunk = data;
            own_chunk_size
                = own_chunk_size + received_chunk_size;
        }
    }
 
    // Stop the timer
    time_taken += MPI_Wtime();
 
    // Opening the other file as taken form input
    // and writing it to the file and giving it
    // as the output
    if (rank_of_process == 0) {
 
        // Printing total number of elements
        // in the file
        printf(
            "Total number of Elements in the array : %d\n",
            own_chunk_size);

 
        // For Printing in the terminal
        printf("Total number of Elements given as input : "
               "%d\n",
               number_of_elements);

        
        printf("Sorted: ");
        CALI_MARK_BEGIN(correctness_check);
        bool sort = isSorted(data, number_of_elements);
        CALI_MARK_END(correctness_check);
        if (sort) {
            printf("True\n");
        } else {
            printf("False\n");
        }

        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "QuickSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", number_of_elements); // The number of elements in input dataset (1000)
        adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", number_of_process); // The number of processors (MPI ranks)
        // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
        // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
        adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
 
        printf(
            "\n\nQuicksort %d ints on %d procs: %f secs\n",
            number_of_elements, number_of_process,
            time_taken);
    }
 
    MPI_Finalize();
    return 0;
}


