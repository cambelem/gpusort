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

#define NUMBER_OF_PROCESSORS 1024
#define BLOCK_DIM 256

//prototype for pivot counting kernal
__global__ void d_count_kernel(unsigned int * d_pivots,
  unsigned int * r_buckets, int pivotsLength, unsigned int * r_indices,
  unsigned int * r_sublist, unsigned int * d_in, int itemCount);

//prototype for bucketsort
__global__ void d_bucketsort(unsigned int * d_in, unsigned int * d_indices,
    unsigned int * d_sublist, unsigned int * r_outputlist,
    unsigned int * d_bucketoffsets, int itemCount);

//prototype for the kernel
__global__ void d_sort_kernel();

/*d_crack
*
* Sets up and calls the kernal to brute-force a password hash.
*
* @params
*   hash    - the password hash to brute-force
*   hashLen - the length of the hash
*   outpass - the result password to return
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

    //Find min and max
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

    //Compute pivots through linear interpolation
    unsigned int pivotsLength = (NUMBER_OF_PROCESSORS * 2) - 1;
    unsigned int * pivots = new unsigned int[pivotsLength];
    unsigned int * buckets_count = new unsigned int[pivotsLength];
    int slope = (max - min)/pivotsLength;
    for (unsigned int i = 0; i < pivotsLength; i++) {
      pivots[i] = (slope * i);
      buckets_count[i] = 0;
    }

    /****************************STEP 1****************************************/
    //Launch a kernal to count the number of items in each bucket so we can
    //refine our pivots later.

    //Input/output mallocs
    unsigned int * d_pivots;
    CHECK(cudaMalloc((void**)&d_pivots, pivotsLength * sizeof(unsigned int)));
    unsigned int * r_buckets;
    CHECK(cudaMalloc((void**)&r_buckets, pivotsLength * sizeof(unsigned int)));
    unsigned int * d_in;
    CHECK(cudaMalloc((void**)&d_in, length * sizeof(unsigned int)));
    unsigned int * r_indices;
    CHECK(cudaMalloc((void**)&r_indices, length * sizeof(unsigned int)));
    unsigned int * r_sublist;
    CHECK(cudaMalloc((void**)&r_sublist,
      (pivotsLength + 1) * sizeof(unsigned int)));

    //Copying things to memory
    CHECK(cudaMemcpy(d_pivots, pivots,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in, in,
      length * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r_buckets, buckets_count,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));

    //kernel dimensions
    dim3 block(BLOCKDIM, 1, 1);
    dim3 grid(ceil((float) length/BLOCKDIM), 1, 1);

    //Launching kernel
    d_count_kernel<<<grid, block>>>(d_pivots, r_buckets, pivotsLength,
      r_indices, r_sublist, d_in, length);

    CHECK(cudaDeviceSynchronize());

    unsigned int * buckets = (unsigned int *) Malloc(pivotsLength * sizeof(unsigned int));
    CHECK(cudaMemcpy(buckets, r_buckets, pivotsLength * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int * indices = (unsigned int *) Malloc(length * sizeof(unsigned int));
    CHECK(cudaMemcpy(indices, r_indices, length * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int * sublist = (unsigned int *) Malloc(length * sizeof(unsigned int));
    cudaMemcpy(sublist, r_sublist, length * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    CHECK(cudaFree(d_pivots));
    CHECK(cudaFree(r_buckets));
    CHECK(cudaFree(r_indices));
    CHECK(cudaFree(r_sublist));
    CHECK(cudaFree(d_in));

    // unsigned int sum = 0;
    // for (unsigned int i = 0; i < pivotsLength; i++) {
    //   if (i % 10 == 0) {
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::setw(8) << buckets[i] << ", ";
    //   sum += buckets[i];
    // }
    // std::cout << std::endl;
    //
    // std::cout << std::endl << sum << " == " << length << std::endl;

    /***************************STEP 1 COMPLETE********************************/

    // std::cout << std::endl;
    //
    // std::cout << "Before" << std::endl;
    // for (unsigned int i = 0; i < 20; i++) {
    //   std::cout << pivots[i] << std::endl;
    // }

    /***************************STEP 2*****************************************/
    // buckets is our count per bucket
    // indices is, for each item, the count of the bucket it was placed in, before it was placed there.
    // sublist is the bucket in which a given item was placed.
    unsigned int N = length;
    unsigned int L = NUMBER_OF_PROCESSORS * 2;
    unsigned int elemsneeded = N/L;

    for (unsigned int i = 0; i < L - 1; i++) {
      int range = pivots[i + 1] - pivots[i];
      while (buckets[i] >= elemsneeded) {
        pivots[i + 1] += (elemsneeded/buckets[i]) * range;
        elemsneeded = N/L;
        buckets[i] -= elemsneeded;
      }
      elemsneeded -= buckets[i];
      pivots[i + 1] += range / 2;
    }
    /*****************************STEP 2 COMPLETE******************************/

    // std::cout << "After" << std::endl;
    // for (unsigned int i = 0; i < 20; i++) {
    //   std::cout << pivots[i] << std::endl;
    // }
    //
    // int count = 0;
    // for (int i = 0; i < length; i++) {
    //   if (in[i] >= pivots[8] && in[i] <= pivots[9]) {
    //     count++;
    //   }
    // }
    // std::cout << "CPU Count: " << count << std::endl;

    /****************************STEP 3****************************************/
    //Launch a kernal to count the number of items in each bucket after
    //redefining pivots!

    //Copying things to memory
    //Input/output mallocs
    CHECK(cudaMalloc((void**)&d_pivots, pivotsLength * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**)&r_buckets, pivotsLength * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**)&d_in, length * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**)&r_indices, length * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**)&r_sublist,
      (pivotsLength + 1) * sizeof(unsigned int)));

    //Copying things to memory
    CHECK(cudaMemcpy(d_pivots, pivots,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in, in,
      length * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r_buckets, buckets_count,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));

    //Launching kernel
    d_count_kernel<<<grid, block>>>(d_pivots, r_buckets, pivotsLength,
      r_indices, r_sublist, d_in, length);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(buckets, r_buckets, pivotsLength * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(indices, r_indices, length * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaMemcpy(sublist, r_sublist, length * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    CHECK(cudaFree(d_pivots));
    CHECK(cudaFree(r_buckets));
    CHECK(cudaFree(r_indices));
    CHECK(cudaFree(r_sublist));
    CHECK(cudaFree(d_in));

    min = UINT_MAX;
    int min_index = 0;
    for (int i = 0; i < length; i++) {
      if (in[i] < min) {
        min = in[i];
        min_index = i;
      }
    }
    std::cout << min << " at index " << min_index << std::endl;
    std::cout << "pivot index = " << sublist[min_index] << std::endl;
    std::cout << "pivot value = " << pivots[sublist[min_index]] << std::endl;
    std::cout << "pivot index = " << indices[min_index] << std::endl;

    free(pivots);

    // int sum = 0;
    // for (unsigned int i = 0; i < pivotsLength; i++) {
    //   if (i % 10 == 0) {
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::setw(8) << buckets[i] << ", ";
    //   sum += buckets[i];
    // }
    // std::cout << std::endl;
    //
    // std::cout << std::endl << sum << " == " << length << std::endl;

    /***************************STEP 3 COMPLETE********************************/

    //Calculate prefix sums for buckets to find the starting index of each
    //bucket in our final bucketsorted array.
    unsigned int * prefix_buckets = (unsigned int *) Malloc(pivotsLength * sizeof(unsigned int));
    prefix_buckets[0] = buckets[0];
    for (unsigned int i = 1; i < pivotsLength; i++) {
      prefix_buckets[i] = prefix_buckets[i - 1] + buckets[i - 1];
    }

    // for (unsigned int i = 0; i < 30; i++) {
    //   if (i % 10 == 0) {
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::setw(8) << buckets[i] << ", ";
    // }
    // std::cout << std::endl;
    //
    // for (unsigned int i = 0; i < 30; i++) {
    //   if (i % 10 == 0) {
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::setw(8) << prefix_buckets[i] << ", ";
    // }
    // std::cout << std::endl;

    /***********************STEP 4: BUCKETSORT*********************************/

    CHECK(cudaMalloc((void**)&d_in, length * sizeof(unsigned int)));
    unsigned int * d_indices;
    CHECK(cudaMalloc((void**)&d_indices, length * sizeof(unsigned int)));
    unsigned int * d_sublist;
    CHECK(cudaMalloc((void**)&d_sublist, length * sizeof(unsigned int)));
    unsigned int * r_outputlist;
    CHECK(cudaMalloc((void**)&r_outputlist, length * sizeof(unsigned int)));
    unsigned int * d_bucketoffsets;
    CHECK(cudaMalloc((void**)&d_bucketoffsets, pivotsLength * sizeof(unsigned int)));

    CHECK(cudaMemcpy(d_in, in,
      length * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_indices, indices, length * sizeof(unsigned int),
            cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sublist, sublist, length * sizeof(unsigned int),
            cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bucketoffsets, prefix_buckets,
      pivotsLength * sizeof(unsigned int), cudaMemcpyHostToDevice));

    d_bucketsort<<<grid, block>>>(d_in, d_indices, d_sublist,
      r_outputlist, d_bucketoffsets, length);

    CHECK(cudaDeviceSynchronize());

    unsigned int * outputlist = (unsigned int *) Malloc(length * sizeof(unsigned int));
    CHECK(cudaMemcpy(outputlist, r_outputlist, length * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    max = 0;
    for (unsigned int i = prefix_buckets[8]; i < prefix_buckets[10]; i++) {
      if (outputlist[i] > max) {
        max = outputlist[i];
      }
      // if (i % 10 == 0) {
      //   std::cout << std::endl;
      // }
      // std::cout << std::setw(8) << outputlist[i] << ", ";
    }

    min = UINT_MAX;
    for (unsigned int i = prefix_buckets[10]; i < prefix_buckets[15]; i++) {
      if (outputlist[i] < min) {
        min = outputlist[i];
      }
    }

    //TODO: Buckets are not working properly.  Elements are not mapping to the
    //correct bucket, nor are they ending up in the right index.  

    std::cout << max << " < " << min << std::endl;
    std::cout << std::endl;

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_indices));
    CHECK(cudaFree(d_sublist));
    CHECK(cudaFree(r_outputlist));
    CHECK(cudaFree(d_bucketoffsets));
    free(outputlist);

    free(buckets);
    free(indices);
    free(sublist);

    // unsigned long size = 2 * NUMCHARS * sizeof(unsigned char);
    // unsigned long outsize = pow(NUMCHARS, 2) * 3;

    // unsigned char * d_passwords;
    // CHECK(cudaMalloc((void**)&d_passwords, size));
    // unsigned char * d_result;
    // CHECK(cudaMalloc((void**)&d_result, outsize));


    //Copy the starting passwords array and valid characters to the GPU
    // CHECK(cudaMemcpyToSymbol(VALID_CHARS, VALID_CHARS_CPU, NUMCHARS * sizeof(char)));
    // CHECK(cudaMemcpy(d_passwords, STARTING_PASSES, 2 * NUMCHARS, cudaMemcpyHostToDevice));


    // Beginning of Four-way Radix Sort
    // We need a block size of 256
    // dim3 block(BLOCKDIM, 1, 1);
    // dim3 grid(1, 1, 1);

    // d_generate_kernel<<<grid, block>>>(d_passwords, 1, NUMCHARS, d_result);

    // CHECK(cudaDeviceSynchronize());
    //
    // unsigned char * passwords = (unsigned char *) Malloc(outsize);
    // CHECK(cudaMemcpy(passwords, d_result, outsize, cudaMemcpyDeviceToHost));
    //
    // CHECK(cudaFree(d_passwords));
    // CHECK(cudaFree(d_result));
    //
    // free(passwords);

    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    //calculate the elapsed time between the two events
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

__global__ void d_count_kernel(unsigned int * d_pivots,
  unsigned int * r_buckets, int pivotsLength, unsigned int * r_indices,
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

__global__ void d_bucketsort(unsigned int * d_in, unsigned int * d_indices,
    unsigned int * d_sublist, unsigned int * r_outputlist,
    unsigned int * d_bucketoffsets, int itemCount) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < itemCount) {
        int newpos = d_bucketoffsets[d_sublist[idx]] + d_indices[idx];
        r_outputlist[newpos] = d_in[idx];
      }
}

/*d_generate_kernel
*  Kernal code executed by each thread to generate a list of all possible
*  passwords of length n + 1.  To do this, each thread will work on one element
*  in passwords and append all characters in VALID_CHARS to it. This kernal
*  works in place, so it will alter the input array.
*
*  @params:
*   passwords - array filled with current passwords to build off of.
*   length    - length of the given passwords
*   n         - number of items currently in passwords array
*   d_result  - location to place newly generated passwords.
*/
__global__ void d_sort_kernel() {
  // unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
}
