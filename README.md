# Hybrid Sort

Hybrid sort is a partial implementation of Sintorn and Assarsson's suggested hybrid GPU sorting algorithm for CUDA outlined in their paper: "Fast parallel GPU-sorting using a hybrid algorithm".  This code is our best interpretation of the algorithm outlined in their original paper.  

# Building Hybrid Sort

We're using a makefile to build Hybrid Sort.  You'll need a device with a GPU that supports CUDA.  You also need to have CUDA's GCC/G++ installed in order to build this application.  To execute the makefile, simply enter the project directory and run `make`!

# Running Hybrid Sort

Usage:

``hybrid_sort lower_bound upper_bound``

In order to execute the Hybrid Sort routine, you need to execute the hybrid_sort executable built by our makefile and pass as input an upper and lower bound for the size of input to generate and sort.  The input upper and lower bounds are entered as powers of 2, so inputting 14 as a lower bound tells the application to sort 2^14 randomly generated input elements.  The application will randomly generate an input data set using rand().  If you called hybrid sort using `hybrid_sort 15 16`, the hybrid sort routine would randomly generate an input of size 2^15 and then call the sorting routine twice, keeping the result from the second run.  Then it would randomly generate an input of size 2^16 and repeat the sorting.  The application will output a string telling the input size and the number of milliseconds it took to sort the data.  

## On Testing Methodologies

For each given input size, we call the sorting kernel twice and keep the second of the two attempts.  Nvidia recommends this testing methodology because the first launch of a kernel on the GPU is typically a "cold" launch, and after the first execution the application's data is usually cached.  The performance of the second execution is indicative of the performance of all subsequent calls to the same routine.  Basically, executing it twice and keeping the second result is the tried and true method of getting consistent results, because of CUDA's caching behavior.  

## Concerning MergeSort

We attempted to implement a standard mergesort on the GPU, but were unable to get the applicaiton after 40+ hours of debugging.  We opted to include a basic n^2 sort instead, which works properly on the GPU, as part of the demo code.  The code for mergesort is still included in d_hybrid_sort.cu and you can, by commenting and uncommenting lines in d_sort_kernel in d_hybrid_sort.cu, activate the mergesort.  The mergesort implementation does not sort the input data properly, but it executes the same number of operations as a working standard mergesort would, so its performance should be indicative of mergesort's performance.  

# Files Overview

This application consists of 2 primary files.

* hybrid_sort.cu - contains the driver code for the application.  This processes command-line arguments and generates random input data, then calls the sorting subroutine the proper number of times and processes output.  
* d_hybrid_sort.cu - contains the sorting subroutine in d_sort and then the necessary kernels to count items in each bucket, execute a bucketsort, and then execute a sequential sort or a mergesort.  The code is fully commented so you can take a look through it if you want!  
