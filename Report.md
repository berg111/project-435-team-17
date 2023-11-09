# CSCE 435 Group project

## 1. Group members:
1. Jake Bergin
2. Jace Thomas
3. Cameron Hoholik-Carlson
4. Ethan McKinney

---

## Team Communication
Team communication will take place through texts and Slack.

## 2. _due 10/25_ Project topic
Sorting.

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

- Merge Sort (MPI + CUDA)

  Merge sort is a comparison-based algorithm that uses a divide-and-conquer approach for sorting arrays. To make parallel, we use the sequential algorithm but scatter the array equally across multiple processes.
  
  Psuedocode (source: https://en.wikipedia.org/wiki/Merge_sort):
  ```
  function merge_sort(list m) is
    // Base case. A list of zero or one elements is sorted, by definition.
    if length of m ≤ 1 then
        return m

    // Recursive case. First, divide the list into equal-sized sublists
    // consisting of the first half and second half of the list.
    // This assumes lists start at index 0.
    var left := empty list
    var right := empty list
    for each x with index i in m do
        if i < (length of m)/2 then
            add x to left
        else
            add x to right

    // Recursively sort both sublists.
    left := merge_sort(left)
    right := merge_sort(right)

    // Then merge the now-sorted sublists.
    return merge(left, right)

  function merge(left, right) is
    var result := empty list

    while left is not empty and right is not empty do
        if first(left) ≤ first(right) then
            append first(left) to result
            left := rest(left)
        else
            append first(right) to result
            right := rest(right)

    // Either left or right may have elements left; consume them.
    // (Only one of the following loops will actually be entered.)
    while left is not empty do
        append first(left) to result
        left := rest(left)
    while right is not empty do
        append first(right) to result
        right := rest(right)
    return result
  ```

- Odd-Even Transposition Sort (openMP + CUDA)

  Odd-Even Sort is a compare and exchange algorithm, that compares odd and even pairs, and after n phases all of the elements will be sorted, and is availabe in parallelism.

  Pseudocode (source: CSCE 435 Slide Deck "07_CSCE_435_algorithms.pdf slide 52"
  ```
  procedure ODD-EVEN_PAR (n) 
  
  begin 
     id := process's label 
  	
     for i := 1 to n do 
     begin 
  	
        if i is odd and id is odd then 
           compare-exchange_min(id + 1); 
        else 
           compare-exchange_max(id - 1);
  			
        if i is even and id is even then 
           compare-exchange_min(id + 1); 
        else 
           compare-exchange_max(id - 1);
  			
     end for
  	
  end ODD-EVEN_PAR
  ```

- Odd Even Sort (MPI)

  Pseudocode source: https://www.dcc.fc.up.pt/~ricroc/aulas/1516/cp/apontamentos/slides_sorting.pdf

  ```
  rank = process_id();
  A = initial_value();
  for (i = 0; i < N; i++) {
    if (i % 2 == 0) { // even phase
      if (rank % 2 == 0) { // even process
        recv(B, rank + 1); send(A, rank + 1);
        A = min(A,B);
    } else { // odd process
      send(A, rank - 1); recv(B, rank - 1);
      A = max(A,B);
  }
  } else if (rank > 0 && rank < N - 1) { // odd phase
      if (rank % 2 == 0) { // even process
        recv(B, rank - 1); send(A, rank - 1);
        A = max(A,B);
    } else { // odd process
        send(A, rank + 1); recv(B, rank + 1);
        A = min(A,B);
    }
}

  ```
  
- Bucket Sort (MPI)

  Bucket Sort is a sorting algorithm that splits each element into different "buckets" based on the number of elements being sorted.
  Each bucket is then sorted using insertion sort.
  After each bucket is sorted, the buckets are stitched back together into one sorted array.
  This algorithm has a time complexity of O(n^2).

  Pseudocode (source: ~):
  ```
  procedure bucketSortMPI()
    A : list of sortable items
    n := length(A)
    buckets : vector of n float arrays
    
    MPI_Init()
    MPI_Comm_rank(taskid)
    MPI_Comm_size(numTasks)

    if master then
        // initialize data
        initializeData(A);
    
        // put elements into buckets
        for i := 0 to n-1 inclusive do
            buckets[n*A[i]] = A[i]
        end for
    
        // send buckets to worker tasks
        for i := 0 to numTasks-1 inclusive do
            MPI_Send(buckets[i])
        end for
  
        // receive sorted buckets from tasks
        for i := 0 to numTasks-1 inclusive do
          MPI_Recv(buckets[i])
        end for
    
        // stitch buckets into one sorted array
        index := 0
        for i := 0 to n-1 inclusive do
            for j := 0 to buckets[i].size()-1 inclusive do
                A[i] = buckets[i][j]
                index++
            end for
        end for
  
        // check for correctness
        correctnessCheck()
  
    if worker then
        // receive bucket from master task
        MPI_Recv(bucket)
        
        // run insertion sort on bucket
        insertionSort(bucket)
  
        // send bucket back to master task
        MPI_Send(bucket)

    // Calculate min, max, and average times
    MPI_Reduce()


    // Calculate times
  end procedure
  ```

- Quick Sort (MPI)

  Quick Sort is a sorting algorithm that that operates in a "divide and conquer" method.
  It takes in the array of data and chooses a starting element, then sorts the array into two sides, elements that are smaller than the starting element and elements that are larger than the starting element.
  Once done, it will take the first half of thenewly ordered set of data and will choose a new starting element, then repeats the step of sorting them to the left and right of the new starting element absed on if the piece of data is smaller or larger.
  This continues until the algorithm is down to 2 elements and then works its way back up the partitioned sets of data.

  Pseudocode (source: https://en.wikipedia.org/wiki/Quicksort):
  ```
  // Sorts a (portion of an) array, divides it into partitions, then sorts those
  algorithm quicksort(A, lo, hi) is 
  // Ensure indices are in correct order
  if lo >= hi || lo < 0 then 
    return
    
  // Partition array and get the pivot index
  p := partition(A, lo, hi) 
      
  // Sort the two partitions
  quicksort(A, lo, p - 1) // Left side of pivot
  quicksort(A, p + 1, hi) // Right side of pivot

  // Divides array into two partitions
  algorithm partition(A, lo, hi) is 
  pivot := A[hi] // Choose the last element as the pivot

  // Temporary pivot index
  i := lo - 1

  for j := lo to hi - 1 do 
    // If the current element is less than or equal to the pivot
    if A[j] <= pivot then 
      // Move the temporary pivot index forward
      i := i + 1
      // Swap the current element with the element at the temporary pivot index
      swap A[i] with A[j]

  // Move the pivot element to the correct pivot position (between the smaller and larger elements)
  i := i + 1
  swap A[i] with A[hi]
  return i // the pivot index
  ```

### 2c. Evaluation plan - what and how will you measure and compare
- Varying array sizes of integers (100, 1000, 10000, 100000, 500000). (Floats for bucket sort impl).
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
- Number of threads in a block on the GPU

Comparisons will be done between different implementations and performance for varying parameters will be examined for each.
