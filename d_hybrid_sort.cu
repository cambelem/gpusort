#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <openssl/md5.h>
#include "d_hybrid_sort.h"
#include "CHECK.h"
#include "config.h"
#include "wrappers.h"

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
float d_sort() {

    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use cuda functions to do the timing
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));

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
