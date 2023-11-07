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

void oddEvenSort(std::vector<int> &array){
    int size = array.size();
    bool sorted = false;

    while (sorted == false){
        sorted = true;

        //#pragma omp parallel for shared(array, sorted)
        //Even phase
        for(int i = 1; i < size - 1; i += 2){
            if (array[i] > array[i + 1]){
                std::swap(array[i], array[i + 1]);
                sorted = false;
            }
        }

        //#pragma omp parallel for shared(array, sorted)
        //Odd phase
        for(int i = 0; i < size - 1; i += 2){
            if (array[i] > array[i + 1]){
                std::swap(array[i], array[i + 1]);
                sorted = false;
            }
        }



    }
}

int main() {

    std::vector<int> array = {10, 9, 8, 12, 5, 7};

    oddEvenSort(array);

    std::cout << "Sorted Array: ";
    for (int num : array) {
        std::cout << num << " ";
    }
    std::cout << std::endl;


}