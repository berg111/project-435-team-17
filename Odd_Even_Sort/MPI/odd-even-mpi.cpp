#include "mpi.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> 
using namespace std;

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0           /* taskid of the first task */
#define FROM_MASTER 1      /* setting a message type for sending data to workers */
#define FROM_WORKER 2      /* setting a message type for receiving data from master */


void generate_array(int *array, int size, int input) {
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
            int temp[size];
            for (int i = 0; i < size; i++) {
                temp[i] = i;
            }
            for(int i = 0; i < size; ++i){
                array[i] = temp[size - 1 - i];
            }
            break;
    }

}

bool isSorted(int *array, int size){
    // cout<< "Size: " << size << endl;
    // for(int i=0;i<size;i++){
    //      	printf("%d ",array[i] );
    //      }
    for(int i = 0; i < size - 1; ++i){
        if(array[i] > array[i + 1]){
            cout << array[i] << " " << array[i + 1];
            return false;
        }
    }
    return true;
}


void oddEvenSort(int array[], int size) {
    int phase, i, temp;

    for (phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) {
            for (i = 1; i < size; i += 2) {
                if (array[i] < array[i - 1]) {
                    temp = array[i];
                    array[i] = array[i - 1];
                    array[i - 1] = temp;
                }
            }
        } else {
            for (i = 1; i < size - 1; i += 2) {
                if (array[i] > array[i + 1]) {
                    temp = array[i];
                    array[i] = array[i + 1];
                    array[i + 1] = temp;
                }
            }
        }
    }
}

int main(int argc, char *argv[]){

    CALI_CXX_MARK_FUNCTION;
    const char* main = "main";
    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comm_small = "comm_small";
    const char* comm_large = "comm_large";
    const char* comp = "comp";
    const char* comp_small = "comp_small";
    const char* comp_large = "comp_large";
    const char* correctness_check = "correctness_check";

	int nump,rank;
	int n,localn;
	int *data,*recdata,*recdata2;
	int *temp;
	int ierr,i;
	int root_process;
	MPI_Status status;
	std::string inputType;
	ierr = MPI_Init(&argc, &argv);
    double rank_time = MPI_Wtime();
    root_process = 0;
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &nump);
    MPI_Comm newcomm;
    MPI_Comm_split(MPI_COMM_WORLD, rank > 0, rank - 1, &newcomm);
    cali::ConfigManager mgr;
    mgr.start();

      if(rank == root_process) {
        n = atoi(argv[1]);
        int input = atoi(argv[2]);
         int avgn = n / nump;
         localn=avgn;
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

    	data=(int*)malloc(sizeof(int)*n);

        CALI_MARK_BEGIN(data_init);
        generate_array(data, n, input);
        CALI_MARK_END(data_init);
        //  printf("array data is:");
        //  for(i=0;i<n;i++){
        //  	printf("%d ",data[i] );
        //  }
        //  printf("\n");
    }
    else{
    	data=NULL;
    }

    

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    ierr=MPI_Bcast(&localn,1,MPI_INT,0,MPI_COMM_WORLD);
    recdata = (int *)malloc(sizeof(int) * localn);
    recdata2 = (int *)malloc(sizeof(int) * localn);
    ierr=MPI_Scatter(data, localn, MPI_INT, recdata, localn, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    sort(recdata,recdata+localn);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);


    //begin the odd-even sort
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int oddrank,evenrank;

    if(rank%2==0){
    	oddrank=rank-1; 
    	evenrank = rank+1;
	}
 	else {
 		oddrank = rank+1;
 		evenrank = rank-1;
	}

    if (oddrank == -1 || oddrank == nump)
    oddrank = MPI_PROC_NULL;
    if (evenrank == -1 || evenrank == nump)
    evenrank = MPI_PROC_NULL;
    
    int p;
    for (p=0; p<nump-1; p++) {
    if (p%2 == 1) /* Odd phase */
        MPI_Sendrecv(recdata, localn, MPI_INT, oddrank, 1, recdata2,
        localn, MPI_INT, oddrank, 1, MPI_COMM_WORLD, &status);
    else /* Even phase */
        MPI_Sendrecv(recdata, localn, MPI_INT, evenrank, 1, recdata2,
        localn, MPI_INT, evenrank, 1, MPI_COMM_WORLD, &status);

    temp=(int*)malloc(localn*sizeof(int));
    for(i=0;i<localn;i++){
        temp[i]=recdata[i];
    }
    if(status.MPI_SOURCE==MPI_PROC_NULL)	continue;
    else if(rank<status.MPI_SOURCE){
        //store the smaller of the two
        int i,j,k;
        for(i=j=k=0;k<localn;k++){
            if(j==localn||(i<localn && temp[i]<recdata2[j]))
                recdata[k]=temp[i++];
            else
                recdata[k]=recdata2[j++];
        }
    }
    else{
        //store the larger of the two
        int i,j,k;
        for(i=j=k=localn-1;k>=0;k--){
            if(j==-1||(i>=0 && temp[i]>=recdata2[j]))
                recdata[k]=temp[i--];
            else
                recdata[k]=recdata2[j--];
        }
    }//else
    }//for

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);


    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    ierr=MPI_Gather(recdata,localn,MPI_INT,data,localn,MPI_INT,0,MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    if(rank==root_process){
    CALI_MARK_BEGIN(correctness_check);
    
        bool sorted = isSorted(data, n);
        cout << "Sorted? " << sorted << endl;
    
    CALI_MARK_END(correctness_check);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Odd Even Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", nump); // The number of processors (MPI ranks)
        // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
        // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 17); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online & Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    }
     rank_time = MPI_Wtime() - rank_time;

    double rank_time_max,
      rank_time_min,
      rank_time_sum,
      rank_time_average;

    MPI_Reduce(&rank_time, &rank_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, newcomm);
    MPI_Reduce(&rank_time, &rank_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, newcomm);
    MPI_Reduce(&rank_time, &rank_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, newcomm);

    if (rank == 0) {

        MPI_Recv(&rank_time_max, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rank_time_min, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rank_time_average, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout << "Rank Time Max = " << rank_time_max << endl;
        cout << "Rank Time Min = " << rank_time_min << endl;
        cout << "Rank Time Avg = " << rank_time_average << endl;

        adiak::value("rank_time_max", rank_time_max);
        adiak::value("rank_time_min", rank_time_min);
        adiak::value("rank_time_average", rank_time_average);


    } else if (rank == 1) {
        rank_time_average = rank_time_sum / (double)(nump - 1);

        MPI_Send(&rank_time_max, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rank_time_min, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rank_time_average, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    }
    free(recdata2);
    free(recdata);
    ierr = MPI_Finalize();


mgr.stop();
mgr.flush();

}

// int main(int argc, char** argv) {
//     CALI_CXX_MARK_FUNCTION;
//     int numtasks, taskid, numworkers, source, dest, mtype;
//     int size = atoi(argv[1]);
//     int arrayINput = atoi(argv[2]);
//     int array[size];
//     MPI_Status status;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
//     MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

//     if (numtasks < 2) {
//         printf("Need at least two MPI tasks. Quitting...\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//         exit(1);
//     }

//     numworkers = numtasks - 1;

//     MPI_Comm new_comm;
//     MPI_Comm_split(MPI_COMM_WORLD, task_id > 0, task_id - 1, &newcomm);

//     double whole_start_time = MPI_Wtime();
//     double recv_total_time, send_total_time, calc_total_time = 0;



//     MPI_Finalize();
//     return 0;
// }
