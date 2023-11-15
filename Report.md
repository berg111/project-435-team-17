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

  MPI_Scatter(arr, range, MPI_INT, arr_copy, range, MPI_INT, 0, MPI_COMM_WORLD);

	merge_sort(arr_copy, 0, range - 1);

	MPI_Gather(arr_copy, range, MPI_INT, arr, range, MPI_INT, 0, MPI_COMM_WORLD);
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
  
- Bucket Sort (MPI + CUDA)

  Bucket Sort is a sorting algorithm that splits each element into different "buckets" based on the number of elements being sorted.
  Each bucket is then sorted using insertion sort.
  After each bucket is sorted, the buckets are stitched back together into one sorted array.
  This algorithm has a time complexity of O(n^2).

  MPI Pseudocode (source: ~):
  ```
  begin procedure bucketSortMPI()
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

begin procedure bucketSortCUDA()
	BLOCKS : number of blocks used for program
 	THREADS : number of threads used for program
 	
	// Initialize Array
 	A : list of sortable items
  	n := length(A)

	// Initialize Data
 	initializeData(A)

  	// Initialize Buckets
   	Buckets : 2D list of sortable items

     	// n = m
      	m := length(Buckets)
    	n := length(Buckets[])

      	// Fill buckets with null values
       	for i := 0 to n-1 inclusive do
		for j := 0 to m-1 inclusive do
  			buckets[i][j] = -1.0;
     		end for
       	end for

 	// Fill buckets with array values
  	for i := 0 to n-1 inclusive do
		for j := 0 to m-1 inclusive do
  			if buckets[n * A[i]] == -1 then
     				buckets[n * A[i]] = A[i]
	 			break;
     			end if
     		end for
       	end for

 	// Sort each bucket with insertion sort
  	insertionSort<<<BLOCKS, THREADS>>>(buckets, n)

   	// Sync with threads
    	cudaDeviceSynchronize()

      	// Stitch buckets back together
       	index : count starting at 0
       	for i := 0 to n-1 inclusive do
		for j := 0 to m-1 inclusive do
  			if buckets[i][j] == -1 then
	 			break;
     			end if

   			A[index] = buckets[i][j]
      			index++
     		end for
       	end for

 	// Check for correctness
  	correctnessCheck(A, n)
  			

end procedure
```


Quick Sort (MPI & CUDA)

  Quick Sort is a sorting algorithm that that operates in a "divide and conquer" method.
  It takes in the array of data and chooses a starting element, then sorts the array into two sides, elements that are smaller than the starting element and elements that are larger than the starting element.
  Once done, it will take the first half of thenewly ordered set of data and will choose a new starting element, then repeats the step of sorting them to the left and right of the new starting element absed on if the piece of data is smaller or larger.
  This continues until the algorithm is down to 2 elements and then works its way back up the partitioned sets of data.

  Pseudocode (source: ~):
  ```
  Quicksort(array, left, right):
  	if left < right:
  		pivot = array.at(right)
  		index = left - 1
  		for i = 1, i < right:
  			if array.at(i) <= pivot:
  				//swap values to the correct side of the partition
  				swap array.at(index) with array.at(i)

			swap array.at(index + 1) with array.at(right)
			partition = index + 1
  		//recursively sort left side of partition
  		Quicksort(array, left, partition - 1)              //for CUDA Quicksort<<1, 1>>(array, left partition - 1)
  		//recursively sort right side of partition
  		Quicksort(array, partition + 1, right)
  ```

### 2c. Evaluation plan - what and how will you measure and compare
- Varying array sizes of integers (100, 1000, 10000, 100000, 500000). (Floats for bucket sort impl).
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
- Number of threads in a block on the GPU

Comparisons will be done between different implementations and performance for varying parameters will be examined for each.

### 4. 

Odd Even Sort:

	For Odd Even Sort - CUDA, the following array sizes were used: 2^16, 2^18, 2^20, 2^22, 2^24, 2^26
 	and the array types: Random, Sorted, Reverse Sorted

	In the lower portion of the array sizes, even as the thread counts increased, the performance increases were pretty much non-existent, 
	providing minimal time decreases.  
 	it is not until array size 2^20 until performance increases can be visualized. Hoever, the performance gains level off fairly quickly once
  	512 threads is reached. There was also a bug that went unsolved that didn't allow 2048+ threads to properly sort, but due to the plateauing of
   	the values, I don't think these thread counts would have made any improvement. 

    	For specific values, no matter the thread count, every run time for 2^18 was under 1 second, and there were no improvements between thread counts.
     	For 2^18, every run was under 3 seconds, again with no improvements or variability between array types.
      	For 2^20, things began to change, there was a steep drop from 20 seconds to 14 seconds, and then to 10 seconds before leveling off. No changes between 
        array types.
       	For 2^22, There was a drop from 250 seconds to 125, and then to 80. Minimal variablity, but sorted array can be seen to run quicker.
	For 2^24, There is a drop from 3500 seconds to 1700, and then the array types split off here. The random array levels off at about 1600 seconds,
 	the sorted array levels off at about 1200 seconds. And the reverse array levels off at 1800 seconds. And actually at 1024 threads, there is a slight 
 	performance decrease.


  	For Odd Even Sort - OMP, the following array sizes were used: 2^16, 2^18, 2^20, higher sizes were not used due to time constraints and performance drops

	The OMP implementation provided some more clear observations than CUDA. The time differences and peerformance changes are much more clear. 

 	For array size 2^16, times decreased from threads 2 to 16, but began to increase at 64+ threads. There are clear differences in the arrays, the sorted
  	arrays provide the lowest times and the random times provide the highest, and reverse is in the middle.

 	For array 2^18, the performance trends are the same as the array prior, however, as the threads increased, the performance drop off was not as severe as 
  	the smaller array

   	For array size 2^20, again the same performnce trends hold true, except the random arrays for the lower thread counts take much longer compared to the 
    	sorted arrays, for example the sorted array for 8 threads took 73 seconds, where the random array took 253 seconds.

	Overall, the general performance for OMP was worse compared to CUDA, however the efficiency for the threads was much greater for OMP, and visualizing
 	the data was much easier and provided greater insights.
     
    	
   
  	
	
  
