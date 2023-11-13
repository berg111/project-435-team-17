#include <iostream>
#include <string.h>
#include <random>
#include <math.h>
#include <vector>

using std::cout, std::endl, std::string, std::vector;

void initializeData(int numDecimalPlaces, float *array, int size) 
{
    long long int randMax = pow(10, numDecimalPlaces);

    for (int i = 0; i < size; i++) {
        array[i] = (float)((rand() % (randMax - 1)) + 1) / (float)randMax;
    }
}

void insertionSort(float *array, int size) 
{
    int index;
    float temp;

    for (int i = 1; i < size; i++) {
        // Set index
        index = i;

        // Loop through elements behind current element
        while ((array[index] < array[index - 1]) && (array[index] != -1.0)) {
            // Swap elements
            temp = array[index];
            array[index] = array[index - 1];
            array[index - 1] = temp;

            // Change index
            index--;
            
            // End condition
            if (index == 0) {
                break;
            }
        }
    }
}

void bucketSort(float *array, int size)
{
    // Sets index
    int index = 0;

    // Creates buckets
    float** buckets = (float**)malloc(size * sizeof(float*));

    for (int i = 0; i < size; i++) {
        buckets[i] = (float*)malloc(size * sizeof(float));
    }

    // Filling buckets with null values
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            buckets[i][j] = -1.0;
        }
    }

    // Pushes array elements into buckets
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (buckets[(int)(size * array[i])][j] == -1.0) {
                buckets[(int)(size * array[i])][j] = array[i];
                break;
            }
        }
    }

    // Sorts each bucket with insertion sort
    for (int i = 0; i < size; i++) {
        insertionSort(buckets[i], size);   
    }

    // Stitches each bucket back together
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (buckets[i][j] == -1) {
                break;
            }

            array[index] = buckets[i][j];
            index++;
        }
    }
}

bool correctnessCheck(float *array, int size) 
{
    for (int i = 1; i < size; i++) {
        if (array[i] < array[i - 1]) {
            return false;
        }
    }

    return true;
}


int main() 
{      
    // Initializes array to hold data
    int size = 15;
    float* array = (float*)malloc(size * sizeof(float));

    // Initializes data to be sorted
    initializeData(6, array, size);

    // Prints unsorted array
    cout << "Array before being sorted:" << endl;
    for (int i = 0; i < size; i++) {
        cout << "array[" << i << "] = " << array[i] << endl;
    }
    cout << endl;

    // Calls bucket sort
    bucketSort(array, size);

    // Checks correctness of sorted array
    bool test = correctnessCheck(array, size);

    if (test) {
        cout << "Array is correctly sorted." << endl;
    } else {
        cout << "Array is sorted incorrectly." << endl;
        cout << endl;
        cout << "Array after being sorted:" << endl;
        for (int i = 0; i < size; i++) {
            cout << "array[" << i << "] = " << array[i] << endl;
        }
    }


    return 0;
}