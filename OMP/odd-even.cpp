#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
// #include <caliper/cali.h>
// #include <adiak.hpp>

void oddEvenSort(std::vector<int> &array, int threads){
    int size = array.size();
    int i = 0;
    int phase = 0;
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

void generate_array(std::vector<int>& array, int size)
{
    array.resize(size);

	for(int i = 0; i < size; i++){
		array[i] = rand() % 100;
    }
}

int main(int argc, char** argv) {

        

    int threads = atoi(argv[1]);
    int arraySize = atoi(argv[2]);

    omp_set_num_threads(threads);

    std::cout << "threads";

    std::vector<int> array;

    generate_array(array, arraySize);

    oddEvenSort(array, threads);

    std::cout << "Sorted Array: ";
    for (int i = 0; i < array.size(); ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;



}