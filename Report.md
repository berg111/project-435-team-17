# CSCE 435 Final Project: Group 17

## 1. Group Members:
1. Jake Bergin
2. Jace Thomas
3. Cameron Hoholik-Carlson
4. Ethan McKinney

### Team Communication Methods
Team communication will take place through texts and Slack.

---

## 2. Project Topic: Comparing Parallel Sorting Algorithms

### 2a. Brief Project Description
- Over the course of this report, our team will analyze the performance results of 4 different parallel sorting algorithms on two different architectures:
	1) Merge Sort
		- MPI
  		- CUDA 
	3) Odd Even Sort
		- MPI
  		- CUDA
 	4) Bucket Sort
		- MPI
  		- CUDA
 	5) Quick Sort 
		- MPI
  		- CUDA

### 2b. Pseudocode for Parallel Algorithms

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

  MPI Pseudocode (source: Handwritten):
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
  CUDA Pseudocode (source: Handwritten):
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


- Quick Sort (MPI & CUDA)

  Quick Sort is a sorting algorithm that that operates in a "divide and conquer" method.
  It takes in the array of data and chooses a starting element, then sorts the array into two sides, elements that are smaller than the starting element and elements that are larger than the starting element.
  Once done, it will take the first half of thenewly ordered set of data and will choose a new starting element, then repeats the step of sorting them to the left and right of the new starting element absed on if the piece of data is smaller or larger.
  This continues until the algorithm is down to 2 elements and then works its way back up the partitioned sets of data.

  Pseudocode (source: Handwritten):
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

### 2c. Evaluation Plan
The following list outlines what we will measure and compare for each algorithm as well has how we will accomplish this:
- Varying array sizes of integers (2^16 - 2^28). (Floats for bucket sort impl).
	- NOTE: There may be small variations in input sizes per algorithm depending on the memory limitations of each implementation 
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
- Number of threads in a block on the GPU

Comparisons will be done between different implementations and performance for varying parameters will be examined for each.

## 3. Implementation
### 3a. Caliper Instrumentation and Calltrees
The following are the calltrees implemented during the instrumentation of each parallel algorithm:
- Merge Sort:

- Odd Even Sort:

- Bucket Sort:

- Quick Sort:


## 4. Performance Analysis
#### Merge Sort

As the input size increases, so does the overall runtime of the program. This is an intuitive result that isn’t all that surprising. 
What is interesting is the relationship between the number of processes and ‘main’ runtime for a fixed input size. For a fixed input
size, as the number of processes increases, so does the runtime of the ‘main’ function. This is not what we had hoped to see because it
demonstrates that our program does not scale well with the number of processes. After seeing the results, I believe that this is caused
by the fact that in order for Merge Sort to work, we must merge many sorted arrays. The number of processes is proportional to the
number of mergers that must be done. In this particular program, as we increase the number of processes, we are able to split the
original array into a larger number of  smaller arrays. As a result, we must spend more time merging arrays. Another interesting
observation that can be made by looking at the output graphs is the rate at which the runtime increases as the number of processes
increases. Not only does the runtime get worse as the number of processes increases, but the rate of increase seems to be higher
for larger input sizes. This can be seen in the following plots.

![mpi main 65536 random](./MergeSort/mpi/report_images/main-a65536-random.png)
![mpi main 1048576 random](./MergeSort/mpi/report_images/main-a1048576-random.png)
![mpi main 4194304 random](./MergeSort/mpi/report_images/main-a4194304-random.png)


From the plots we can see that the curves are steeper for larger problem sizes. I believe this supports the idea that the increased 
amount of merging is drastically slowing down the program. One thing to note is that the problem sizes shown here may not 
be giving the implementation an opportunity to shine. As we increase the problem size and number of processes, we give 
the downward trend of the 'comp' region an opportunity to more heavily influence the overall runtime of 'main'. The plateau seen 
in the 'comm' region would also help to improve the performance at higher input sizes with more processes.

For different input types, the runtime of ‘main’ didn’t seem to be affected. The plots definitely vary between the input types; 
however, the time spent within the program is approximately the same. Here are the plots used:

![mpi main 4194304 random](./MergeSort/mpi/report_images/main-a4194304-random.png)
![mpi main 4194304 perturbed](./MergeSort/mpi/report_images/main-a4194304-perturbed.png)
![mpi main 4194304 reverse](./MergeSort/mpi/report_images/main-a4194304-reverse.png)
![mpi main 4194304 sorted](./MergeSort/mpi/report_images/main-a4194304-sorted.png)

For the smaller number of processes, the curves seem to be very similar (except for the ‘reverse’ case). The peaks of the graphs 
all occur around the same 60-80 second mark near the larger number of processes. I believe that if multiple runs were performed, 
we would see an average case for all input types that is more similar. For three of the plots, the runs with 32 processes seem 
to run for around 30 seconds; however, for the ‘reverse’ case this is not true. This could be due to the fact that more swaps 
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
threads; however, the ‘total GPU time’ increased uniformly as the number of threads increased. If the lines in the plots 
had been flat, we could conclude that the program has good weak scaling. In this case, we can conclude the opposite because 
the lines appear to be increasing exponentially. This shows that the program is not handling the larger problem sizes well 
even when the number of threads is increased to account for the difference. One potential issue with this test is that the 
relationship between the number of threads and problem size was not linear. For this part of the experiment, the problem 
size was equal to the number of threads squared. If instead we used a constant factor to relate the number of threads and 
the size of the input, the plots may have been flatter and suggested better weak scaling performance.

![mpi and cuda speedup](./MergeSort/mpi/report_images/speedup.png)

We can see that the speedup for CUDA is less uniform than the speedup for MPI. Both implementations have poor 
parallel performance. Neither reach a speedup that indicates peformance better than the sequential implementation. 
There is a relationship between input size and speedup for the CUDA implementation. It seems from these plots that 
if we were to increase the size of the problem beyond 16777216, we may see the speedup go beyond 1 (indicating that 
it is performing better than the sequential version). For MPI, the speed immediately drops off as the number of 
processes increases. There is no notable difference between the input sizes and types. This suggests that the 
implementation is performing badly. There does not seem to be any indication that there exists a combination of 
processes and input size where the speedup goes beyond 1.


#### Odd Even Sort:

For Odd Even Sort - CUDA, the following array sizes were used: 2^16, 2^18, 2^20, 2^22, 2^24, 2^26
and the array types: Random, Sorted, Reverse Sorted, and 1 % perturbed and thread counts: 64, 128, 256, 512, 1024.

The performance on the CUDA implemntation scaled very well, and shows a solid trend in decreasing time as threads increase. The communication overhead does take some time on lower thread counts, but quickly drops off. The speedup time also shows great performance and increases until about 512 threads and tapers off. Overall, the CUDA implementation seemed to perform much more efficiently than the MPI implementation did.

![pic](./Odd_Even_Sort/CUDA-Speedup.png)
![pic](./Odd_Even_Sort/CUDA-Comm.png)
![pic](./Odd_Even_Sort/CUDA-Comp.png)
![pic](./Odd_Even_Sort/CUDA-Main.png)
![pic](./Odd_Even_Sort/CUDA-Weak.png)

For Odd Even Sort - MPI, the following array sizes were used: 2^16, 2^18, 2^20, 2^22, 2^24, 2^26
and the array types: Random, Sorted, Reverse Sorted, and 1 % perturbed and thread counts: 2, 4, 8, 16, 32, 64, 128, 256, 512.

In the lower end of the Input sizes, this algorithm does not scale well in MPI. As the processes increase, so does the time taken, regardless of the input type
It can be noted that in the comm time, that there is a spike in the mid range of processes, and this is due to the number of nodes used in the job file, where only
1 was being utilized until about 64 processes. This spike could be due to the nature of the communication method, which in this algorithm I utilized
broadcast, scatter, and gather communications. On weak scaling, it scales relatively well on the lower end of processes, where the processes are handling roughly
the same load, until about 512 processes where it begins to increase. The speedup of the lower input also has an inverse trend, showing a poor performance trend.

![pic](./Odd_Even_Sort/MPI-speedup.png)
![pic](./Odd_Even_Sort/MPI-comm.png)
![pic](./Odd_Even_Sort/MPI-comp.png)
![pic](./Odd_Even_Sort/MPI-main.png)
![pic](./Odd_Even_Sort/MPI-weak.png)

Performance gains become to be noticeable once we reach higher input sizes of about 2^24, where the computation time begins to decrease as the processes increase; however, the same communication overhead still holds true in this input as well. The speedup of this input size scaled better than the lower input but is not great, and requires the processes to be at about 32 to start seeing improvements, that taper off pretty quick.

![pic](./Odd_Even_Sort/MPI-Speedup-2.png)
![pic](./Odd_Even_Sort/MPI-comm-2.png)
![pic](./Odd_Even_Sort/MPI-comp-2.png)
![pic](./Odd_Even_Sort/MPI-Main-2.png)
![pic](./Odd_Even_Sort/MPI-weak.png)

#### Bucket Sort

Two architectures were used to implement bucket sort for this project: CUDA and MPI. Two versions of each were instrumented and used for this analysis. The first set of implementations were handwritten, and the second set of implementations were found online and edited for the purposes of this project.

Starting with the handwritten implementations, 4 different array sizes were used for CUDA: 2^16, 2^18, 2^20, and 2^22 values each. Each of these arrays were input into the sorting algorithm randomized, pre-sorted, reverse sorted, and 1% perturbed. For each of these sizes and organizations, different numbers of thread counts were used on the NVIDIA A100 GPU including 64, 128, 256, 512, and 1024 threads. Unfortunately, further testing of this implementation of Bucket Sort on the Grace hardware was not possible as we ran into a memory allocation limit. Traditional implementations of Bucket Sort utilize vectors to serve as “buckets” for array values to be sorted into; however, this traditional approach is not possible with CUDA. CUDA does not support a dynamic vector data structure, meaning that the size of each bucket must be known when allocating memory which goes against the nature of a memory efficient Bucket Sort algorithm. Our solution to this was to limit the number of buckets to the max thread count (1024) and create a (b * n) array that allows for all possible combinations of buckets. At the max array size, this resulted in a buckets object being created with 1024 * 2^22 * 4 = ~17 GB where 1024 is the number of buckets, 2^22 is the array size, and 4 is the size of the data type we were sorting (floats). If we increased the array size to 2^24, then we would need ~68 GB of memory. Since the A100 GPU only has 40GB of onboard memory, we would either need to use multiple GPUs (which would utilize MPI and CUDA at the same time, defeating the purpose of comparing the two architectures) or we would need different hardware. 

Due to the large overhead of creating and sorting the array into buckets (which was not efficiently parallelizable), the performance of this algorithm gave some interesting numbers. To begin, the ‘Comp Small’ caliper region was used to measure the kernel call to the GPU while the ‘Comp Large’ caliper region was used to measure sorting the array into buckets (preprocessing) and stitching the sorted buckets back together. In the graphs below, you can see the overhead for preprocessing and stitching dominated over the time it took to sort the individual buckets. 

![cuda comp small region](./BucketSort/CUDA/Plots/Random_Comp_Small_Plot.png)
![cuda comp large region](./BucketSort/CUDA/Plots/Random_Comp_Large_Plot.png)

Due to this dominance, this CUDA implementation of Bucket sort saw marginal performance increases as the number of threads was increased. It is also interesting to note that this is consistent across the different array organizations. Below are ‘Random’ and ‘Sorted’ plots for reference.

![cuda random comp small region](./BucketSort/CUDA/Plots/Random_Comp_Small_Plot.png)
![cuda sorted comp small region](./BucketSort/CUDA/Plots/Sorted_Comp_Small_Plot.png)

The next handwritten implementation of Bucket Sort utilized C++ MPI. This version was also limited by memory for the same design reasons as CUDA. 4 different array sizes were used: 2^14, 2^16, 2^22, and 2^20 values each. Each of these arrays were input into the sorting algorithm randomized, pre-sorted, reverse sorted, and 1% perturbed. For each of these sizes and organizations, different numbers of processors were used on multiple Grace nodes including 2, 4, 8, 16, 32, 64, 128, 256, and 512 processors. Unfortunately, further testing of this implementation was not possible as any size larger than 2^20 would crash the algorithm, despite increasing the memory allocation per node on Grace. This is likely because of the use of MPI_Scatter and MPI_Gather. Since only the master task would allocate the initial bucket’s array, each task would have to allocate enough memory for each of the buckets they receive. This effectively results in the ‘buckets’ object being allocated twice which needs lots of memory at large array size. 
Although this version of the algorithm utilized MPI_Scatter and MPI_Gather to make communication more efficient, it is interesting to note that as the number of processors increased, the amount of communication time also increased exponentially. Similar to the CUDA implementation, this algorithm had a dominating region, but it was the ‘Comm’ region rather than the ‘Comp Large’ region. It seems that the computation on each task was MUCH faster on MPI, but the communication was also much slower resulting in degraded performance as the number of processors increased. Below are plots of the ‘Comm’ and ‘Comp Small’ for the MPI implementation. 

![mpi comm region](./BucketSort/MPI/Plots/Random_Comm_Plot.png)
![mpi comp small region](./BucketSort/MPI/Plots/Random_Comp_Small_Plot.png)

Comparing the overall runtimes of both handwritten algorithms, it is easy to see that the communication overheads dominated their runtimes resulting in faster computation as the number of processors/threads increased, but worse total runtimes as the number of processors/threads increased. Despite MPI sorting each bucket faster, it was bar far the slower algorithm. Below are plots of CUDA and MPI ‘Main’ region times. 

![cuda main region](./BucketSort/CUDA/Plots/Random_Main_Plot.png)
![mpi main region](./BucketSort/MPI/Plots/Random_Main_Plot.png)

Since the handwritten bucket sort algorithms were so inefficient and did not yield any performance increase, a second set of implementations were instrumented and run on the Grace cluster. These new implementations were significantly more memory efficient, which fixed the input size limitations of the handwritten versions. 

Starting with the online CUDA implementation, input sizes ranged from 2^16 to 2^22 and thread counts ranged from 64 to 1024. This input range was used because of time limitations with this implementation. 2^24 input values and up took astronomically high times to compute (over 5 hours for 2^24) for the lower thread counts, so it was not reasonable to run those sizes on the Grace cluster. It is worthy of note that this implementation COULD run for sizes of 2^24 to 2^28, so there was not a memory limitation like the handwritten version. The highest input size (2^24) was run with randomized, sorted, reverse sorted, and 1% perturbed inputs while the rest of the input sizes were run with only randomized inputs. As shown in the graphs below, the performance for this implementation increased as the number of threads increased for all input sizes. This performance can also be seen in the speed up graph, as each input size increases in speedup as the number of threads increases.

![cuda main 65536](./BucketSort/Report-Plots/CUDA-Main-65536-1.png)
![cuda main 4194304](./BucketSort/Report-Plots/CUDA-Main-4194304-1.png)
![cuda speedup random](./BucketSort/Report-Plots/CUDA-Random-Speedup-1.png)
 
When comparing different input types rather than input sizes, the same performance trends can be seen across almost all types. The graphs below show that performance increases for each input type across the total main time and the computation regions. The smallest performance increase comes from the 1% perturbed arrays, while the other input types have roughly the same performance. This performance trend can also be seen in the speed up graph below. ‘Random’, ‘Reverse Sorted’, and ‘Sorted’ input types each have similar speed ups, while the ‘1% perturbed’ trend has a much smaller speedup.
   
![cuda comp inputs](./BucketSort/Report-Plots/CUDA-Comp-Inputs-1.png)
![cuda main inputs](./BucketSort/Report-Plots/CUDA-Main-Inputs-1.png)
![cuda inputs speedup](./BucketSort/Report-Plots/CUDA-Inputs-Speedup-1.png)

Moving on to the online MPI implementation, input sizes ranged from 2^16 to 2^28 and processor counts ranged from 2 to 1024. Each of these inputs were given to the sorting algorithm randomized, pre-sorted, reverse sorted, and 1% perturbed. As shown in the graphs for ‘Random’ inputs below, the performance generally increased as the number of processors increased. For lower input sizes, the performance increased initially, but eventually decreased as the communication overhead started to dominate; however, the performance eventually continuously increased as higher input sizes were used. 

![mpi comp large random](./BucketSort/Report-Plots/MPI-Comp-Large-Random-1.png)
![mpi main random](./BucketSort/Report-Plots/MPI-Main-Random-1.png)
![mpi speedup 65536](./BucketSort/Report-Plots/MPI-Speedup-65536-1.png)
![mpi speedup 268435456](./BucketSort/Report-Plots/MPI-Speedup-268435456-1.png)
  
As the speed up graphs show, this algorithm does scale well for higher input sizes. The ‘Random’ input type had the best performance increases, while the other input types eventually tapered off around 64 processes and eventually lost performance due to communication overhead. 
Weak scaling also shows that this algorithm scales well for both architectures. As shown in the graphs below, the weak scaling trends are roughly flat, which is the desired behavior. This flat trend means that as the number of processes/threads increase with the size of the input, each process/thread still has roughly the same amount of work to do. 

![mpi weak scaling](./BucketSort/Report-Plots/MPI-Weak-Scaling-1.png)
![cuda weak scaling](./BucketSort/Report-Plots/CUDA-Weak-Scaling-1.png)


#### Quicksort

For testing the performance of our Quicksort algorithm implemented in CUDA, it was tested on a 2^16, 2^18, 2^20, 2^22, 2^24, 2^26, and 2^28 array sizes with them being sorted in order, sorted randomly, and sorted in reverse.
It was expected that as during the lower thread counts on the smaller arrays, there would not be much difference in the time, yet as the array size grew and the algorithm was run on more threads, we would theoretically see a large reduction in the time taken to complete the sorting process.
Since CUDA functions are inline, there is no way for them to be called recursively inside of themselves. Because of this, the implementation is an attempted work around to try to run the quicksort algorithm on CUDA. 
For our implementation, it was noticeable that for the smallest array size there was virtually no difference in the runtime based on the number of threads that the program was run on. As the array size continued to grow, there was an obvious growth in the amount of time the quicksort algorithm took to finish. Along with this, there was no noticeable difference in the time the algorithm took based on the variation in the number of threads.

![CUDA base processes runtime](./Quicksort/Picture1.png)

For the randomly sorted array we can see that the runtimes, as the array size increased, was increasing at an exponential rate, as beyond the 2^22 array size the algorithm would max out the alloted run time. Once increasing the runtime, we would be able to see some results; however, it would increase at an exponential rate. 

It is evident that in our workaround attempt, there is only one call to the quicksort algorithm that does not correctly allocate equal work to separate threads in the GPU. Therefore, the work was being done on one thread alone and as a result was constantly increasing in time as the array size generated, yet staying virtually the same as the thread count was increased.

For testing the performance of our Quicksort algorithm implemented in MPI, we tested it on the same array size over 2, 4, 8, 16, 32, 64, 128, 256, 512, and 1024 processes. This was tested against arrays sorted in order, sorted randomly, and sorted in reverse.
Per our expectations, as the array size increased, the time to sort would also increase, as this is an obvious observation, and also as the thread count increased for each array size, that array would take less time to sort (i.e. becoming more efficient).
As a note, this implementation involved a standard quicksort algorithm yet at the end of each quicksort step, once the array was split on the partition, the two array sides would then be placed in chunks and wait for a process to pick them up and continue the operation until the bottom level of the array, then pushing all the sorted chunks back together on the way back up the tree. This is a different implementation than that of Merge Sort, as Merge Sort generally sorts the arrays on the way back up the tree, meaning the array is split down the middle over and over until we reach the bottom level of the tree and sort as we merge the arrays back together. Quick Sort involves sorting the array on the way down and reconnecting the ends of each array piece to form the final sorted array.

![MPI comp size 1048576](./Quicksort/MPI/comp.png)
![MPI comm size 1048576](./Quicksort/MPI/comm.png)
![MPI main size 1048576](./Quicksort/MPI/main.png)

From the "comp" graph we can see that the computation time across all array types (i.e. sorted, unsorted, reversed, 1%perturbed) were all extremely similar to each other. The graph shows that the algorithm scaled very well with respect to the increase in the number of processes. For the 2^20 graph that we are seeing there is not much more benefit after 2^5 processes which is where this array size flattens out. 

For our "comm" graph we can see that there was a slight communication overhead when it came to the 1%perturbed array. The flux in the later part of the graph may show that I was running the 1%perturbed array with a high number of processes but the incorrect number of nodes, which would explain a slight flux in the communication; however, in the middle processes which is where we see the graph flatten out in our computation, we can see all array sort types performed very similarly.

For the "main" graph we can see that it is extremely similar to the "comp" graph explained above in that all array types perform the exact same and the algorithm scales particularly well.

![MPI weak scaling random](./Quicksort/MPI/weak.png)
![MPI speed up size 1048576](./Quicksort/MPI/speed.png)

We see in the weak scaling that it appears to trend upward linearly; however, looking closer we can identify that all weak scaling values are under .003 seconds which is very good. And next to the speed up graph looks like there was an astronomical improvement in the run times for this implementation. The 2 process 2^20 array size time was around 386 seconds, and 2^5 processes at the same array size time was around 2 seconds. So we are able to show that there was a substantial improvement in the runtimes; however, it is noted that the speedup may be erroneous.

Overall, I believe it is evident that the Quicksort implementation, although an edited version, since Quicksort is sequential in itself, scaled extremely well as the number of processes increased yet overall was not the fastest algorithm that could have been used.
