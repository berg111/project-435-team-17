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

For example:
- Algorithm 1a (MPI + CUDA)
- Algorithm 1b (MPI on each core)
- Algorithm 2a (MPI + CUDA)
- Algorithm 2b (MPI on each core)

- Merge Sort (MPI)

  Merge sort is a comparison-based algorithm that uses a divide-and-conquer approach for sorting arrays.
  
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

- SECOND ALGO HERE

- Bubble Sort (MPI)

  Bubble Sort is a sorting algorithm that repeatedly compares adjacent elements in an array, moving forward in the array window until it reaches the end of the array.
  If two elements that are being compared have the second element as larger than the first, then those elements are swapped.
  In its most basic and non- optimized form, it repeats this n - 1 times where n is the size of the array. This sorting algorithm is also known as sinking sort.

  Pseudocode (source: https://en.wikipedia.org/wiki/Bubble_sort):
  ```
  procedure bubbleSort(A : list of sortable items)
    n := length(A)
    repeat
        swapped := false
        for i := 1 to n-1 inclusive do
            { if this pair is out of order }
            if A[i-1] > A[i] then
                { swap them and remember something changed }
                swap(A[i-1], A[i])
                swapped := true
            end if
        end for
    until not swapped
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
