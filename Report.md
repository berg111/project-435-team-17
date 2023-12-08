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
and the array types: Random, Sorted, Reverse Sorted, and 1 % perturbed and thread counts: 64, 128, 256, 512, 1024

The performance on the CUDA implemntation scaled very well, and shows a solid trend in decreasing time as threads increase. The communication overhead does take some time on lower thread counts, but quickly drops off. The speedup time also shows great performance and increases until about 512 threads and tapers off. Overall the CUDA implementation seemed to perform much more efficiently than the MPI implementation did.


For Odd Even Sort - MPI, the following array sizes were used: 2^16, 2^18, 2^20, 2^22, 2^24, 2^26
and the array types: Random, Sorted, Reverse Sorted, and 1 % perturbed and thread counts: 2, 4, 8, 16, 32, 64, 128, 256, 512

In the lower end of the Input sizes, this algorithm does not scale well in MPI. As the processes increase, so does the time taken, regardless of the input type
It can be noted that in the comm time, that there is a spike in the mid range of processes, and this is due to the number of nodes used in the jobfile, where only
1 was being utilized until about 64 processes. This spike could be due to the nature of the communication method, which in this algorithm I utilized
broadcast, scatter, and gather communications. On weak scaling, it scales relatively well on the lower end of processes, where the processes are handling roughly
the same load, until about 512 processes where it begins to increase. The speedup of the lower input also has an inverse trend, showing a poor performance trend.

Performance gains become to be noticeable once we reach higher input sizes of about 2^24, where the computation time begins to decrease as the processes increase.
However, the same communication overhead still holds true in this input as well. The speedup of this input size scaled better than the lower input but is not great, and requires the processes to be at about 32 to start seeing improvements, that taper off pretty quick.

#### MergeSort

As the input size increases, so does the overall runtime of the program. This is an intuitive result that isn’t all that surprising. 
What is interesting is the relationship between the number of processes and ‘main’ runtime for a fixed input size. For a fixed input
size, as the number of processes increases, so does the runtime of the ‘main’ function. This is not what we had hoped to see because it
demonstrates that our program does not scale well with the number of processes. After seeing the results, I believe that this is caused
by the fact that in order for MergeSort to work, we must merge many sorted arrays. The number of processes is proportional to the
number of mergers that must be done. In this particular program, as we increase the number of processes, we are able to split the
original array into a larger number of  smaller arrays. As a result, we must spend more time merging arrays. Another interesting
observation that can be made by looking at the output graphs is the rate at which the runtime increases as the number of processes
increases. Not only does the runtime get worse as the number of processes increases, but the rate of increase seems to be higher
for larger input sizes. This can be seen in the following plots.

![mpi main 65536 random](./MergeSort/mpi/report_images/main-a65536-random.png)
![mpi main 1048576 random](./MergeSort/mpi/report_images/main-a1048576-random.png)
![mpi main 4194304 random](./MergeSort/mpi/report_images/main-a4194304-random.png)


From the plots we can see that the curves are steeper for larger problem sizes. I believe this supports the idea that the increased 
amount of merging is drastically slowing down the program.

For different input types, the runtime of ‘main’ didn’t seem to be affected. The plots definitely vary between the input types, 
however the time spent within the program is approximately the same. Here are the plots used:

![mpi main 4194304 random](./MergeSort/mpi/report_images/main-a4194304-random.png)
![mpi main 4194304 perturbed](./MergeSort/mpi/report_images/main-a4194304-perturbed.png)
![mpi main 4194304 reverse](./MergeSort/mpi/report_images/main-a4194304-reverse.png)
![mpi main 4194304 sorted](./MergeSort/mpi/report_images/main-a4194304-sorted.png)

For the smaller number of processes, the curves seem to be very similar (except for the ‘reverse’ case). The peaks of the graphs 
all occur around the same 60-80 second mark near the larger number of processes. I believe that if multiple runs were performed, 
we would see an average case for all input types that is more similar. For three of the plots, the runs with 32 processes seem 
to run for around 30 seconds. However, for the ‘reverse’ case this is not true. This could be due to the fact that more swaps 
occur during the sorting phase of the program. This would increase the amount of time spent sorting each subarray.

As mentioned previously, increasing the number of processes and keeping the problem size fixed did not scale well. For this 
particular program, strong scaling does not perform well. I believe this is due to the increased number of array merges that must 
happen with an increase in processes. For weak scaling, we can look at how well the program performs when the problem size 
increases with the number of processes.

![mpi main weak scaling random](./MergeSort/mpi/report_images/mpi-weak-scaling.png)

From the graph we can see a runtime that is linear with respect to the input size. For each input size, we used a number of 
processors that was directly proportional to the size of the input. The input size was divided by 32,768 to get the number of 
processors. We can determine that the program does not scale well with weak scaling. We can come to this conclusion because 
the runtime is linear (just like the relationship between the number of processors and the input size).

For CUDA: Here are the resulting plots for the ‘comp’ and ‘main’ region with weak scaling.

![cuda weak scaling comp region](./MergeSort/cuda/report_images/comp-weak-scaling.png)
![cuda weak scaling main region](./MergeSort/cuda/report_images/main-weak-scaling.png)

We can see that the ‘main’ portion of the program saw a decrease in the average time per rank when moving from 64 to 128 
threads. However, the ‘total GPU time’ increased uniformly as the number of threads increased. If the lines in the plots 
had been flat, we could conclude that the program has good weak scaling. In this case, we can conclude the opposite because 
the lines appear to be increasing exponentially. This shows that the program is not handling the larger problem sizes well 
even when the number of threads is increased to account for the difference. One potential issue with this test is that the 
relationship between the number of threads and problem size was not linear. For this part of the experiment, the problem 
size was equal to the number of threads squared. If instead we used a constant factor to relate the number of threads and 
the size of the input, the plots may have been flatter and suggested better weak scaling performance.


**Bucket Sort**

Two different implementations of Bucket Sort were used for this project: a CUDA implementation and an MPI implementation. Starting with the CUDA implementation, 4 different array sizes were used: 2^16, 2^18, 2^20, and 2^22 values each. Each of these arrays were input into the sorting algorithm randomized, pre-sorted, reverse sorted, and 1% perturbed. For each of these sizes and organizations, different numbers of thread counts were used on the NVIDIA A100 GPU including 64, 128, 256, 512, and 1024 threads. Unfortunately, further testing of this implementation of Bucket Sort on the Grace hardware was not possible as we ran into a memory allocation limit. Traditional implementations of Bucket Sort utilize vectors to serve as “buckets” for array values to be sorted into; however, this traditional approach is not possible with CUDA. CUDA does not support a dynamic vector data structure, meaning that the size of each bucket must be known when allocating memory which goes against the nature of a memory efficient Bucket Sort algorithm. Our solution to this was to limit the number of buckets to the max thread count (1024) and create a (b * n) array that allows for all possible combinations of buckets. At the max array size, this resulted in a buckets object being created with 1024 * 2^22 * 4 = ~17 GB where 1024 is the number of buckets, 2^22 is the array size, and 4 is the size of the data type we were sorting (floats). If we increased the array size to 2^24, then we would need ~68 GB of memory. Since the A100 GPU only has 40GB of onboard memory, we would either need to use multiple GPUs (which would utilize MPI and CUDA at the same time, defeating the purpose of comparing the two architectures) or we would need different hardware. 

Due to the large overhead of creating and sorting the array into buckets (which was not efficiently parallelizable), the performance of this algorithm gave some interesting numbers. To begin, the ‘Comp Small’ caliper region was used to measure the kernel call to the GPU while the ‘Comp Large’ caliper region was used to measure sorting the array into buckets (preprocessing) and stitching the sorted buckets back together. In the graphs below, you can see the overhead for preprocessing and stitching dominated over the time it took to sort the individual buckets. 

![cuda comp small region](./BucketSort/CUDA/Plots/Random_Comp_Small_Plot.png)
![cuda comp large region](./BucketSort/CUDA/Plots/Random_Comp_Large_Plot.png)

Due to this dominance, this CUDA implementation of Bucket sort saw marginal performance increases as the number of threads was increased. It is also interesting to note that this is consistent across the different array organizations. Below are ‘Random’ and ‘Sorted’ plots for reference.

![cuda random comp small region](./BucketSort/CUDA/Plots/Random_Comp_Small_Plot.png)
![cuda sorted comp small region](./BucketSort/CUDA/Plots/Sorted_Comp_Small_Plot.png)

The next implementation of Bucket Sort utilized C++ MPI. This version was also limited by memory for the same design reasons as CUDA. 4 different array sizes were used: 2^14, 2^16, 2^22, and 2^20 values each. Each of these arrays were input into the sorting algorithm randomized, pre-sorted, reverse sorted, and 1% perturbed. For each of these sizes and organizations, different numbers of processors were used on multiple Grace nodes including 2, 4, 8, 16, 32, 64, 128, 256, and 512 processors. Unfortunately, further testing of this implementation was not possible as any size larger than 2^20 would crash the algorithm, despite increasing the memory allocation per node on Grace. This is likely because of the use of MPI_Scatter and MPI_Gather. Since only the master task would allocate the initial bucket’s array, each task would have to allocate enough memory for each of the buckets they receive. This effectively results in the ‘buckets’ object being allocated twice which needs lots of memory at large array size. 
Although this version of the algorithm utilized MPI_Scatter and MPI_Gather to make communication more efficient, it is interesting to note that as the number of processors increased, the amount of communication time also increased exponentially. Similar to the CUDA implementation, this algorithm had a dominating region, but it was the ‘Comm’ region rather than the ‘Comp Large’ region. It seems that the computation on each task was MUCH faster on MPI, but the communication was also much slower resulting in degraded performance as the number of processors increased. Below are plots of the ‘Comm’ and ‘Comp Small’ for the MPI implementation. 

![mpi comm region](./BucketSort/MPI/Plots/Random_Comm_Plot.png)
![mpi comp small region](./BucketSort/MPI/Plots/Random_Comp_Small_Plot.png)

Comparing the overall runtimes of both algorithms, it is easy to see that the communication overheads dominated their runtimes resulting in faster computation as the number of processors/threads increased, but worse total runtimes as the number of processors/threads increased. Despite MPI sorting each bucket faster, it was bar far the slower algorithm. Below are plots of CUDA and MPI ‘Main’ region times. 
<INSERT MAIN REGIONS FOR CUDA AND MPI>
![cuda main region](./BucketSort/CUDA/Plots/Random_Main_Plot.png)
![mpi main region](./BucketSort/MPI/Plots/Random_Main_Plot.png)


**Quicksort**

For testing the performance of our Quicksort algorithm implemented in CUDA, it was tested on a 2^16, 2^18, 2^20, 2^22, 2^24, 2^26, and 2^28 array sizes with them being sorted in order, sorted randomly, and sorted in reverse.
It was expected that as during the lower thread counts on the smaller arrays, there would not be much difference in the time, yet as the array size grew and the algorithm was run on more threads, we would theoretically see a large reduction in the time taken to complete the sorting process.
Since CUDA functions are inlined, there is no way for them to be called recursively inside of themselves. Because of this, the implementation is an attempted work around to try to run the quicksort algorithm on CUDA. 
For our implementation, it was noticeable that for the smallest array size there was virtually no difference in the runtime based on the number of threads that the program was run on. As the array size continued to grow, there was an obvious growth in the amount of time the quicksort algorithm took to finish. Along with this, there was no noticeable difference in the time the algorithm took based on the variation in the number of threads.

![CUDA base processes runtime](./Quicksort/Picture1.png)

For the randomly sorted array we can see that the runtimes, as the array size increased, was increasing at an exponential rate, as beyond the 2^22 array size the algorithm would max out the elloted run time. Once increasing the runtime, we would be able to see some results, however it would increase at an exponential rate. 

It is evident that in our workaround attempt, there is only one call to the quicksort algorithm that does not correctly allocate equal work to separate threads in the GPU. Therefore the work was being done on one thread alone and as a result was constantly increasing in time as the array size generated, yet staying virtually the same as the thread count was increased.

For testing the performance of our Quicksort algorithm implemented in MPI, we tested it on the same array size over 2, 4, 8, 16, 32, 64, 128, 256, 512, and 1024 processes. This was tested against arrays sorted in order, sorted randomly, and sorted in reverse.
By our expectations, the MPI implementation was expected to run the longest on the longest array size with the least amount of processes. By our evaluations, this was mostly true, as our implementation revealed that as we increased the array sizes, our implementation was taking longer as expected. However, adding on processes was not having much of an effect on the performance. 

![MPI random sort runtime](./Quicksort/Picture2.png)

For the plot above, we can see that a randomly sorted graph would take longer to sort with an increase in the array size. At a certain array size, 2^22 to be exact, the algorithm would error out or simply not sort the algorithm and claim it was finished. Because of this we assumed there was not enough memory allocated to the program when running on a higher array size. However after finding out that the implementation was sending the full data set to each process, it was evident that the processes were not equally sharing the data for the sorting algorithm and were allocating all the work to a single process, as this is an implementation error that we were coming across. 
  
