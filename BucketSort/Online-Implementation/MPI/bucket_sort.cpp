#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

//////////QuickSort Stuff

// Comparison function used by qsort
int compare_dbls(const void* arg1, const void* arg2){
 double a1 = *(double *) arg1;
 double a2 = *(double *) arg2;
 if (a1 < a2) return -1;
 else if (a1 == a2) return 0;
 else return 1;
}

// Sort the array in place
void qsort_dbls(double *array, int array_len){
 qsort(array, (size_t)array_len, sizeof(double), compare_dbls);
} 

////////////////

//Function to find what bucket a double belongs to based
//off how many processors there are
int find_bucket(double num, int p_num){
  int x;
  for(x=1; x < p_num+1; x++){
	double bucket_range =(double) x / (double)p_num;
	if(num <= bucket_range){
	  return x - 1; //return bucket number
	}
  }
  return -1;
}

//Bucket sort v2 w/ non-uniform distr

int main(int argc, char *argv[]){
    CALI_CXX_MARK_FUNCTION;

    // Define Caliper region names
    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comm_small = "comm_small";
    const char* comm_large = "comm_large";
    const char* comp = "comp";
    const char* comp_small = "comp_small";
    const char* comp_large = "comp_large";
    const char* correctness_check = "correctness_check";

	int myrank, P;
	int sub_count[1];

	if(argc != 2){
		printf("\nPlease include N, problem size\n");
		return 0;
	}

	//Allocate Arrays	
	int N = strtol(argv[1], NULL, 10);

	//Init MPI, get process # and ranks
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);	
	
	double t1 = MPI_Wtime();

	//printf("\nP: %d\n", P);
	
	//Generate N/P Elements on each processor 
	//Generate Array w/ random numbers, and counts of #s for each bucket
	int numpp = N/P;
	double *list = (double*)malloc(numpp*sizeof(double));
	int *count = (int*)calloc(P, sizeof(int));
	int i, j, bucket;
	double r;
    // printf("Generating array on rank %d with %d values\n", myrank, numpp);

    CALI_MARK_BEGIN(data_init);
    // 1% PERTURBED INPUT
    j = 0;
    for(i = myrank * numpp; i < (myrank * numpp) + numpp; i++){

		r = (double)(i) / (double)(N + 1);
		// r = r * r;
		list[j] = r;
		//Determine bucket count to increase
		bucket = find_bucket(r, P);
		count[bucket]++;
        j++;
		//printf("Count of bucket %d: %d, ", bucket, count[bucket]);
	}

    int num_perturbations = numpp / 100;
    int rand_index1, rand_index2;
    double temp;
    if (myrank == 0) {
        printf("num_perturbations per bucket = %d\n", num_perturbations);
    }
    for (int i = 0; i < num_perturbations; i++) {
        rand_index1 = rand() % numpp;
        rand_index2 = rand() % numpp;
        temp = list[rand_index1];
        list[rand_index1] = list[rand_index2];
        list[rand_index2] = temp;
    }

    // REVERSED INPUT
    // j = 0;
    // for(i = (P - myrank) * numpp; i > ((P - myrank) * numpp) - numpp; i--){

	// 	r = (double)(i) / (double)(N + 1);
	// 	// r = r * r;
	// 	list[j] = r;
	// 	//Determine bucket count to increase
	// 	bucket = find_bucket(r, P);
	// 	count[bucket]++;
    //     j++;
	// 	//printf("Count of bucket %d: %d, ", bucket, count[bucket]);
	// }

    // SORTED INPUT
    // j = 0;
    // for(i = myrank * numpp; i < (myrank * numpp) + numpp; i++){

	// 	r = (double)(i) / (double)(N + 1);
	// 	// r = r * r;
	// 	list[j] = r;
	// 	//Determine bucket count to increase
	// 	bucket = find_bucket(r, P);
	// 	count[bucket]++;
    //     j++;
	// 	//printf("Count of bucket %d: %d, ", bucket, count[bucket]);
	// }

    // RANDOM INPUT
	// for(i = 0; i < numpp; i++){

	// 	r = (double)rand() / (double) RAND_MAX;
	// 	r = r * r;
	// 	list[i] = r;
	// 	//Determine bucket count to increase
	// 	bucket = find_bucket(r, P);
	// 	count[bucket]++;
	// 	//printf("Count of bucket %d: %d, ", bucket, count[bucket]);
	// }
    CALI_MARK_END(data_init);

    // if (myrank == 0) {
    //     for(i = 0; i < 10; i++) {
    //         printf("list[%d] = %f\n", i, list[i]);
    //     }
    // }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
	int *bucket_count = (int*)malloc(P*sizeof(int));
	MPI_Alltoall(count, 1, MPI_INT, bucket_count, 1, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

	int loc_bcount = 0;
	//Add together counts
	for(i = 0; i < P; i++){
		loc_bcount+= bucket_count[i]; 
	}

	//Bucket Counts for Debugging
	//printf("\nBUCKET %d, Count: %d\n", myrank, loc_bcount);

	//Allocate arrays based on counts
	double *bucket_list = (double*)malloc(loc_bcount*sizeof(double));

	//Distribute list to other processes
	CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
	//Allocate arrays required for distribution
	int *displs = (int*)malloc(P*sizeof(int));
	double *dist_list = (double*)malloc(numpp*sizeof(double));
	int *index = (int*)calloc(P,sizeof(int));

	//Create Displacements for scatterv & gatherv
	//Send displacements
	displs[0] = 0;
	for(i = 1; i < P; i++){
		displs[i] = count[i-1] + displs[i-1];
	}

	//Receive displacements
	int *rdispls = (int*)malloc(P*sizeof(int));
	rdispls[0] = 0;
	for(i = 1; i < P; i++){
		rdispls[i] = bucket_count[i-1] + rdispls[i-1];
	}

	for(i = 0; i < numpp; i++){
		//Find bucket for double
		bucket = find_bucket(list[i], P);
		//Place double in list
		dist_list[displs[bucket] + index[bucket]] = list[i];
		//update index
		index[bucket]++;
	}
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
	free(list);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
	MPI_Alltoallv(dist_list, count, displs, MPI_DOUBLE, bucket_list, bucket_count, rdispls, MPI_DOUBLE, MPI_COMM_WORLD); 	
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

	//Do Quicksort on each list locally
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
	qsort_dbls(bucket_list, loc_bcount);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

	//Gather counts of each bucket to root
	int gathercounts[1];
	gathercounts[0] = loc_bcount;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
	MPI_Gather(gathercounts, 1, MPI_INT, count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

	//Count at root now holds size each bucket
	if(myrank==0){
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        displs[0] = 0;
        for(i = 1; i < P; i++){
            displs[i] = count[i-1] + displs[i-1];
        }
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);
	}
	
	double* final_list = (double*)malloc(N*sizeof(double));
	//Gather all lists at root 
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
	MPI_Gatherv(bucket_list,loc_bcount, MPI_DOUBLE, final_list, count, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
	double t2 = MPI_Wtime();

	//Check Result
	if(myrank == 0){
        CALI_MARK_BEGIN(correctness_check);
		int sorted = 1;
		int k;
		for(k = 0; k < N - 2; k++){
			if(final_list[k] > final_list[k+1]){
				sorted = 0;
			}
		}
        CALI_MARK_END(correctness_check);

		if(sorted == 1){
			printf("\nSORTING CORRECT\n");
		}else{
			printf("\nSORTING NOT CORRECT\n");
		}
		printf("\n2.1- N: %d P: %d  Execution Time: %f\n", N, P, t2-t1);

        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "BucketSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", N); // The number of elements in input dataset (1000)
        adiak::value("InputType", "1%%perturbed"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", P); // The number of processors (MPI ranks)
        // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
        // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
        adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

	}

	
	//Free allocated Arrays
	free(index);
	free(displs);
	free(rdispls);
	free(count);
	free(bucket_count);
	free(bucket_list);
	free(final_list);
	MPI_Finalize();

}
