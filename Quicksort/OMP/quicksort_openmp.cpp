#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>


void quicksort(std::vector<int> &array, int l, int r) {
    if (l < r) {
        // #pragma omp parallel shared(array, l, r)
        // for (int i = 0; i < 5; i++) {
        //     std::cout << array[i] << " ";
        // }
        // std::cout << std::endl;
        int pivot = array[r];
        int index = l - 1;

        #pragma omp for
        for (int i = l; i < r; i++) {
            if (array[i] <= pivot) {
                index++;
                std::swap(array[index], array[i]);
            }
        }

        std::swap(array[index + 1], array[r]);
        int part = index + 1;

        #pragma omp task
        {
            quicksort(array, l, part - 1);
        }

        #pragma omp task
        {
            quicksort(array, part + 1, r);
        }
        // for (int i = 0; i < 5; i++) {
        //     std::cout << array[i] << " ";
        // }
        // std::cout << std::endl;
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
    int array_size = atoi(argv[2]);

    omp_set_num_threads(threads);

    std::vector<int> array;
    generate_array(array, array_size);

    // std::cout << array_size << std::endl;

    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     quicksort(array, 0, array_size - 1);
    // }
    quicksort(array, 0, array_size - 1);
    std::cout << "Sorted array: ";
    for (int i = 0; i < array_size; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}