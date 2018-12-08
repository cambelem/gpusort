#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <openssl/md5.h>
#include "d_hybrid_sort.h"
#include "CHECK.h"
#include "config.h"
#include "wrappers.h"
#include <algorithm>
#include <functional>
#include <array>

#define NUMBER_OF_PROCESSORS 1024
#define BLOCK_DIM 256

//prototype for pivot counting kernal
__global__ void d_count_kernel(unsigned int * d_pivots,
  int * r_buckets, int pivotsLength, unsigned int * r_indices,
  unsigned int * r_sublist, unsigned int * d_in, int itemCount);

//prototype for bucketsort
__global__ void d_bucketsort(unsigned int * d_in, unsigned int * d_indices,
    unsigned int * d_sublist, unsigned int * r_outputlist,
    unsigned int * d_bucketoffsets, int itemCount);

//prototype for the sequential version of mergesort.  This is basically a
//sequential n^2 mergesort that is used when there are too few buckets remaining
//in order to wrap everything up in as few inefficient steps as possible.
//NOTE: We do not use this function, but we left it in because it still works,
//      its just slower!
__device__ void d_sequential_mergesort(unsigned int * d_in,
  unsigned int * r_output, unsigned int startIndex, unsigned int endIndex);

//prototype for the sorting kernel that will control what mergesort to call.
__global__ void d_sort_kernel(unsigned int * d_in,
  unsigned int * d_bucketoffsets, unsigned int * r_outputlist, int itemCount,
  int bucketsCount);

__device__ void d_mergesort(unsigned int * input, unsigned int * working,
                              unsigned int startIndex, unsigned int endIndex);

__device__ void d_merge(unsigned int * data, unsigned int * working,
                        unsigned int start, unsigned int middle,
                        unsigned int end);

/*d_sort
*
* Sets up and calls the kernal to sort input.
*
* @params
*   in     - the randomly generated input array
*   length - the length of the input array
*/
float d_sort(unsigned int * in, unsigned int length) {

    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use cuda functions to do the timing
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));

    //Find min and max of input data
    unsigned int max = 0;
    unsigned int min = UINT_MAX;
    for (unsigned int i = 0; i < length; i++) {
      if (in[i] < min) {
        min = in[i];
      }
      if (in[i] > max) {
        max = in[i];
      }
    }

    //Compute pivots through linear "interpolation".  Evenly spaced points
    //between the smallest and largest input elements
    unsigned int pivotsLength = (NUMBER_OF_PROCESSORS * 2) - 1;
    unsigned int * pivots = new unsigned int[pivotsLength];
    int * buckets_count = new int[pivotsLength];
    int slope = (max - min)/pivotsLength;
    unsigned int j = 0;
    for (unsigned int i = 0; i < pivotsLength; i++) {
      pivots[i] = (slope * j);
      buckets_count[i] = 0;
      j += length/pivotsLength;
    }


    /****************************STEP 1****************************************/
    //Launch a kernal to count the number of items in each bucket so we can
    //refine our pivots later.

    //Input/output mallocs
    unsigned int * d_pivots;
    CHECK(cudaMalloc((void**)&d_pivots, pivotsLength * sizeof(unsigned int)));
    int * r_buckets;
    CHECK(cudaMalloc((void**)&r_buckets, pivotsLength * sizeof(int)));
    unsigned int * d_in;
    CHECK(cudaMalloc((void**)&d_in, length * sizeof(unsigned int)));
    unsigned int * r_indices;
    CHECK(cudaMalloc((void**)&r_indices, length * sizeof(unsigned int)));
    unsigned int * r_sublist;
    CHECK(cudaMalloc((void**)&r_sublist, length * sizeof(unsigned int)));

    //Copying data to GPU memory
    CHECK(cudaMemcpy(d_pivots, pivots,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in, in,
      length * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r_buckets, buckets_count,
      pivotsLength * sizeof(int), cudaMemcpyHostToDevice));

    //kernel dimensions
    dim3 block(BLOCKDIM, 1, 1);
    dim3 grid(ceil((float) length/BLOCKDIM), 1, 1);

    //Launching kernel
    d_count_kernel<<<grid, block>>>(d_pivots, r_buckets, pivotsLength,
      r_indices, r_sublist, d_in, length);

    //Wait for the kernel to finish
    CHECK(cudaDeviceSynchronize());

    int * buckets = (int *) Malloc(pivotsLength * sizeof(int));
    //Copy the bucket counts back off of the GPU
    CHECK(cudaMemcpy(buckets, r_buckets, pivotsLength * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //Free up unused space on the GPU
    CHECK(cudaFree(r_indices));
    CHECK(cudaFree(r_sublist));

    /***************************STEP 1 COMPLETE********************************/

    std::cout << "Counting with Original Pivots Complete\n";

    /*********************STEP 2: Refining Pivots******************************/
    //Refine our pivots using the algorithm suggested by Sintorn and Assarsson,
    //Or at least our best interpretation of their pseudocode and description!

    // buckets is our count per bucket
    // indices is, for each item, the count of the bucket it was placed in, before it was placed there.
    // sublist is the bucket in which a given item was placed.
    unsigned int N = length;
    unsigned int L = NUMBER_OF_PROCESSORS * 2;
    int elemsneeded = ceil((float) N/L);

    for (unsigned int i = 0; i < pivotsLength - 1; i++) {
      int range = pivots[i + 1] - pivots[i];
      while (buckets[i] >= elemsneeded) {
        pivots[i + 1] += (elemsneeded/buckets[i]) * range;
        elemsneeded = ceil((float) N/L);
        buckets[i] -= elemsneeded;
      }
      elemsneeded -= buckets[i];
      pivots[i + 1] += range / 2;
    }


    /*****************************STEP 2 COMPLETE******************************/

    std::cout << "Done Refining Pivots\n";

    /***************STEP 3: Counting After Refining Pivots*********************/
    //Launch a kernal to count the number of items in each bucket after
    //redefining pivots!

    //Copying data to GPU memory
    //Input/output mallocs
    CHECK(cudaMalloc((void**)&r_indices, length * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**)&r_sublist, length * sizeof(unsigned int)));

    //Copying data to GPU memory
    CHECK(cudaMemcpy(d_pivots, pivots,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r_buckets, buckets_count,
      pivotsLength * sizeof(int), cudaMemcpyHostToDevice));

    //Launching kernel
    d_count_kernel<<<grid, block>>>(d_pivots, r_buckets, pivotsLength,
      r_indices, r_sublist, d_in, length);

    //Wait for the kernel to finish
    CHECK(cudaDeviceSynchronize());

    //Copy the bucket counts back off of the GPU
    CHECK(cudaMemcpy(buckets, r_buckets, pivotsLength * sizeof(int), cudaMemcpyDeviceToHost));

    //Free up unneeded space on the GPU
    CHECK(cudaFree(d_pivots));
    CHECK(cudaFree(r_buckets));

    /***************************STEP 3 COMPLETE********************************/

    std::cout << "Counting with Refined Pivots Complete\n";

    //Calculate prefix sums for buckets to find the starting index of each
    //bucket in our final bucketsorted array.
    unsigned int * prefix_buckets = (unsigned int *) Malloc(pivotsLength * sizeof(unsigned int));
    prefix_buckets[0] = buckets[0];
    for (unsigned int i = 1; i < pivotsLength; i++) {
      prefix_buckets[i] = prefix_buckets[i - 1] + buckets[i - 1];
    }

    /***********************STEP 4: BUCKETSORT*********************************/
    //Launch a kernel to move every element in the input array to its
    //destination bucket.

    //Input/output mallocs
    unsigned int * r_outputlist;
    CHECK(cudaMalloc((void**)&r_outputlist, length * sizeof(unsigned int)));
    unsigned int * d_bucketoffsets;
    CHECK(cudaMalloc((void**)&d_bucketoffsets, pivotsLength * sizeof(unsigned int)));

    //Copying data to GPU memory
    CHECK(cudaMemcpy(d_bucketoffsets, prefix_buckets,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));

    //Launching Kernel
    d_bucketsort<<<grid, block>>>(d_in, r_indices, r_sublist,
      r_outputlist, d_bucketoffsets, length);

    //Wait for the kernel to finish
    CHECK(cudaDeviceSynchronize());

    //Copy the bucketsorted output data back off the GPU.
    unsigned int * outputlist = (unsigned int *) Malloc(length * sizeof(unsigned int));
    CHECK(cudaMemcpy(outputlist, r_outputlist, length * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //Free up unneeded GPU memory.
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(r_indices));
    CHECK(cudaFree(r_sublist));
    CHECK(cudaFree(r_outputlist));
    CHECK(cudaFree(d_bucketoffsets));

    /***********************STEP 4 COMPLETE************************************/

    std::cout << "Done Bucketsorting\n";

    /*************************STEP 5: MERGESORT********************************/
    //Mergesort the bucketsorted output from step 4.

    //Input/output mallocs
    CHECK(cudaMalloc((void**)&d_in, length * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**)&r_outputlist, length * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**)&d_bucketoffsets, pivotsLength * sizeof(unsigned int)));

    //Copying data to the GPU memory
    CHECK(cudaMemcpy(d_in, outputlist,
      length * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r_outputlist, outputlist,
      length * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bucketoffsets, prefix_buckets,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));


    //printf("%d\n", pivotsLength);

    dim3 gridPivotsLength(ceil((float) pivotsLength/BLOCKDIM), 1, 1);

    //Launch a kernel to sort the data.
    d_sort_kernel<<<gridPivotsLength, block>>>(d_in, d_bucketoffsets, r_outputlist, length,
      pivotsLength);

    //Wait for the kernel to finish
    CHECK(cudaDeviceSynchronize());

    //Copy results back from the GPU.
    CHECK(cudaMemcpy(outputlist, r_outputlist, length * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //Free up the rest of our GPU memory.
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(r_outputlist));
    CHECK(cudaFree(d_bucketoffsets));
    //Free up any system memory we don't need anymore.
    free(buckets);
    delete[] pivots;
    delete[] buckets_count;


    /*************************STEP 5 COMPLETE**********************************/

    std::cout << "Finished Sorting\n";

    /*Full output is now in outputlist.  You can print it with by uncommenting
    * the following few lines!
    */
    // for (unsigned int i = 0; i < length; i++) {
    //   std::cout << outputlist[i] << std::endl;
    // }

    //Free the output list once we're done.
    free(outputlist);

    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    //calculate the elapsed time between the two events
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/*d_count_kernel
*
* A kernel to count the number of items in each bucket, find the destination
* bucket for each item, find the index into the destination bucket for each
* item.
*
* This is our interpretation of Sintorn and Assarsson's described algorithm and
*   pseudocode.
*
* @params:
*   d_pivots     - the input array of pivots
*   r_buckets    - a place to return a list of bucket counts
*   pivotsLength - the length of d_pivots and r_buckets
*   r_indices    - a place to return the indices for each item into its
*                   destination bucket
*   r_sublist    - a place to return the bucket each item will map to.
*   d_in         - the input data
*   itemCount    - the length of r_indices, r_sublist, and d_in.
*
* @return:
*   returns passed through pointers passed as input.
*/
__global__ void d_count_kernel(unsigned int * d_pivots,
  int * r_buckets, int pivotsLength, unsigned int * r_indices,
  unsigned int * r_sublist, unsigned int * d_in, int itemCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < itemCount) {
      unsigned int element = d_in[idx];
      unsigned int index = pivotsLength/2 - 1;
      unsigned int jump = pivotsLength/4;
      int pivot = d_pivots[index];
      while(jump >= 1) {
        index = (element < pivot) ? (index - jump) : (index + jump);
        pivot = d_pivots[index];
        jump /= 2;
      }
      index = (element < pivot) ? index : index + 1;
      r_sublist[idx] = index;
      r_indices[idx] = atomicAdd(&r_buckets[index], 1);
    }
}

/*d_bucketsort
*
* A kernel to bucketsort the input elements.  This will simply move elements to
* their destination index.
*
* This is our interpretation of Sintorn and Assarsson's suggested bucketsort
*   taken from their pseudocode and writing.
*
* @params:
*   d_in            - the input data
*   d_indices       - the input array of destination bucket indices for each item
*   d_sublist       - the input array of destination buckets for each item
*   r_outputlist    - the bucketsorted output
*   d_bucketoffsets - an input array with a prefix sums array of the bucket
*                     indices.  This will tell us where each individual bucket
*                     starts.  Used for destination index calculation.
*   itemCount       - the length of all input data.
*
* @return:
*   returns passed through pointers passed as input.
*/
__global__ void d_bucketsort(unsigned int * d_in, unsigned int * d_indices,
    unsigned int * d_sublist, unsigned int * r_outputlist,
    unsigned int * d_bucketoffsets, int itemCount) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < itemCount) {
        int newpos = d_bucketoffsets[d_sublist[idx]] + d_indices[idx];
        r_outputlist[newpos] = d_in[idx];
      }
}

/*d_sequential_mergesort
*
* A kernel to execute a basic n^2 sort on a GPU core.
*
* @params:
*   d_in        - the input data
*   r_output    - space for the output array
*   startIndex  - the index to start sorting at
*   endIndex    - the index to stop sorting at
*
* @return:
*   returns passed through pointers passed as input.
*/
__device__ void d_sequential_mergesort(unsigned int * d_in,
  unsigned int * r_output, unsigned int startIndex, unsigned int endIndex) {
  for (unsigned int i = startIndex; i < endIndex; i++) {
    unsigned int min = UINT_MAX;
    unsigned int min_index = UINT_MAX;
    for (int j = startIndex; j < endIndex; j++) {
      if (d_in[j] < min) {
        min = d_in[j];
        min_index = j;
      }
    }
    r_output[i] = min;
    d_in[min_index] = UINT_MAX;
  }
}

/*d_merge
*
* A traditional merge routine to be executed on the GPU as part of mergesort.
*
* NOT USED.  Originally we wanted to use a standard mergesort on the GPU,
* and originally we thought it worked properly, but due to unpredictable thread
* scheduling and possibly some memory access errors, we've had to disable this
* code in the final demo.  That being said, if you wish to demo the running time
* of a standard mergesort on the GPU, you can uncommend the call to d_mergesort
* in d_sort_kernel and comment out the call to d_sequential_mergesort.  This
* will not successfully sort the input data, but it will execute the same number
* of operations that a working mergesort implementation would, thus its runtime
* should be reflective of actual runtime.
*
* @params:
*   data    - the input data
*   working - working memory to complete the mergesort
*   start   - the start index of the first half of the merge
*   middle  - the middle index dividing the first and second halves of merge
*   end     - the end index of the second half of the array to merge
*
* @return:
*   sorting occurs in place!
*/
__device__ void d_merge(unsigned int * data, unsigned int * working,
                        unsigned int start, unsigned int middle,
                        unsigned int end) {
  unsigned int lower = start;
  unsigned int upper = middle;
  for (unsigned int i = start; i <= end; i++) {
    if (working[lower] < working[upper]) {
      data[i] = working[lower];
      lower++;
    }
    else {
      data[i] = working[upper];
      upper++;
    }
  }
}

/*d_mergesort
*
* A traditional mergesort routine to be executed on the GPU.
*
* NOT USED.  Originally we wanted to use a standard mergesort on the GPU,
* and originally we thought it worked properly, but due to unpredictable thread
* scheduling and possibly some memory access errors, we've had to disable this
* code in the final demo.  That being said, if you wish to demo the running time
* of a standard mergesort on the GPU, you can uncommend the call to d_mergesort
* in d_sort_kernel and comment out the call to d_sequential_mergesort.  This
* will not successfully sort the input data, but it will execute the same number
* of operations that a working mergesort implementation would, thus its runtime
* should be reflective of actual runtime.
*
* @params:
*   input       - the input data
*   working     - working memory to complete the mergesort
*   startIndex  - the start index of the array to mergesort
*   endIndex    - the end index of the array to mergesort
*
* @return:
*   sorting occurs in place!
*/
__device__ void d_mergesort(unsigned int * input, unsigned int * working,
                            unsigned int startIndex, unsigned int endIndex) {
  if (startIndex < endIndex) {
    unsigned int m = startIndex + ((endIndex - startIndex) / 2);
    d_mergesort(input, working, startIndex, m);
    d_mergesort(input, working, m + 1, endIndex);
    d_merge(input, working, startIndex, m, endIndex);
  }
}

/*d_sort_kernel
*  Our "driver" sorting kernel.  This allocates lists to different CUDA cores
*   and launches a mergesort on each bucket.
*
*  @params:
*   d_in - input array
*   d_bucketoffsets - the offsets for the beginning of each bucket in d_in.
*   r_outputlist - working memory for the mergesort!
*   itemCount - the length of d_in and r_outputlist
*   bucketsCount - the length of d_bucketoffsets which is also the number of
*       buckets for our data.
*/
__global__ void d_sort_kernel(unsigned int * d_in,
  unsigned int * d_bucketoffsets, unsigned int * r_outputlist, int itemCount,
  int bucketsCount) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < bucketsCount) {
    d_sequential_mergesort(d_in, r_outputlist, d_bucketoffsets[index], d_bucketoffsets[index + 1]);

    /*If you want to demo the traditional mergesort's performance, you can
    * uncomment the following call to d_mergesort and comment out the
    * d_sequential_mergesort call above.  This will not sort the data but it
    * will do the same number of operations as a working mergesort would,
    * so the performance is comparable.
    */
    // d_mergesort(r_outputlist, d_in, d_bucketoffsets[index], d_bucketoffsets[index + 1]);
  }
}
