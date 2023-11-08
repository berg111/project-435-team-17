#include <iostream>
#include <string.h>
#include <random>
#include <math.h>
#include <vector>

#define SIZE 10

using std::cout, std::endl, std::string, std::vector;

void initializeData(int numDecimalPlaces, float(*array)[SIZE]) 
{
    long long int randMax = pow(10, numDecimalPlaces);

    for (int i = 0; i < SIZE; i++) {
        (*array)[i] = (float)((rand() % (randMax - 1)) + 1) / (float)randMax;
    }
}

void insertionSort(vector<float, std::allocator<float>> *array) 
{
    int index;
    float temp;

    for (int i = 1; i < (*array).size(); i++) {
        // Set index
        index = i;

        // Loop through elements behind current element
        while ((*array)[index] < (*array)[index - 1]) {
            // Swap elements
            temp = (*array)[index];
            (*array)[index] = (*array)[index - 1];
            (*array)[index - 1] = temp;

            // Change index
            index--;
            
            // End condition
            if (index == 0) {
                break;
            }
        }
    }
}

void bucketSort(float(*array)[SIZE])
{
    // Sets index and buckets vector
    int index = 0;
    vector<float> buckets[SIZE];

    // Pushes array elements into buckets
    for (int i = 0; i < SIZE; i++) {
        buckets[(int)(SIZE * (*array)[i])].push_back((*array)[i]);
    }

    // Sorts each bucket with insertion sort
    for (int i = 0; i < SIZE; i++) {
        insertionSort(&(buckets[i]));   
    }

    // Stitches each bucket back together
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < buckets[i].size(); j++) {
            (*array)[index] = buckets[i][j];
            index++;
        }
    }
}

bool correctnessCheck(float (*array)[SIZE]) 
{
    for (int i = 1; i < SIZE; i++) {
        if ((*array)[i] < (*array)[i - 1]) {
            return false;
        }
    }

    return true;
}


int main() 
{      
    // Initializes array to hold data
    float array[SIZE];

    // Initializes data to be sorted
    initializeData(6, &array);

    // Prints unsorted array
    cout << "Array before being sorted:" << endl;
    for (int i = 0; i < SIZE; i++) {
        cout << "array[" << i << "] = " << array[i] << endl;
    }
    cout << endl;

    // Calls bucket sort
    bucketSort(&array);

    // Checks correctness of sorted array
    bool test = correctnessCheck(&array);

    if (test) {
        cout << "Array is correctly sorted." << endl;
        // Prints sorted array
        // cout << "Array after being sorted:" << endl;
        // for (int i = 0; i < SIZE; i++) {
        //     cout << "array[" << i << "] = " << array[i] << endl;
        // }
    } else {
        cout << "Array is sorted incorrectly." << endl;
    }


    return 0;
}