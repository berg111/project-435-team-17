#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

void quicksort(std::vector<int>& array, int l, int r) {
    if (l < r) {
        int pivot = array[l];
        int i = l, j = r;

        while (i < j) {
            while (i < j && array[j] >= pivot)
                j--;

            if (i < j)
                array[i++] = array[j];

            while (i < j && array[i] <= pivot)
                i++;

            if (i < j)
                array[j--] = array[i];
        }

        array[i] = pivot;
        quicksort(array, l, i - 1);
        quicksort(array, i + 1, r);
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
    MPI_Init(&argc, &argv);

    int threads = atoi(argv[1]);
    int array_size = atoi(argv[2]);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> array;
    generate_array(array, array_size);

    if (rank == 0) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        for (int i = 0; i < n; i++) {
            array[i] = std::rand() % 100;
        }
    }

    MPI_Bcast(array.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size = n / size;
    int local_start = rank * local_size;
    int local_end = local_start + local_size;

    quicksort(array, local_start, local_end - 1);

    std::vector<int> sorted_array(n);
    MPI_Gather(array.data() + local_start, local_size, MPI_INT, sorted_array.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        quicksort(sorted_array, 0, n - 1);
        for (int i = 0; i < n; i++) {
            std::cout << sorted_array[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}
