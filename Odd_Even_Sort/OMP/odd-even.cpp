#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <caliper/cali.h>
#include <adiak.hpp>
using namespace std::chrono;
void oddEvenSort(std::vector<int> &array, int threads){
    // CALI_CXX_MARK_FUNCTION;
    int size = array.size();

    #pragma omp parallel shared(array,size)
    for (size_t phase = 0; phase < size; phase++){

        //Even phase
        if(phase % 2 == 0){
            #pragma omp for
            for(int i = 1; i < size - 1; i += 2){
                if (array[i] > array[i + 1]){
                    std::swap(array[i], array[i + 1]);

                }
            }
        }
        
        //Odd phase
        else{
            #pragma omp for
            for(int i = 0; i < size - 1; i += 2){
                if (array[i] > array[i + 1]){
                    std::swap(array[i], array[i + 1]);

                }
            }
        }
    }
}

//Helper function to create array of random values
void generate_array(std::vector<int>& array, int size){
    
    array.resize(size);

	for(int i = 0; i < size; i++){
		array[i] = rand() % 100000;
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

int main(int argc, char** argv) {

    CALI_CXX_MARK_FUNCTION;
    //initialize threads and array size
    int threads = atoi(argv[1]);
    int arraySize = atoi(argv[2]);
    omp_set_num_threads(threads);

    std::cout << "threads";

    std::vector<int> array;
    
    //create array with random values based on size input
    generate_array(array, arraySize);
    auto start = high_resolution_clock::now();
    oddEvenSort(array, threads);

    auto stop = high_resolution_clock::now();

    std::cout << "Sorted Array: ";
    for (int i = 0; i < array.size(); ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
    auto duration = duration_cast<milliseconds>(stop - start);
    if(isSorted(array)){
            std::cout << "Sorted correctly" << std::endl;
    }
    else{
        std::cout << "Failed" << std::endl;
    }
    std::cout << "Time taken by function: "
         << duration.count() / 1000.0 << " seconds" << std::endl;

    aidak::value("threads", threads);
    aidak::value("time_taken", duration/1000.0);

}