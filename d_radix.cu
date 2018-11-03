#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <openssl/md5.h>
#include "d_radix.h"
#include "CHECK.h"
#include "config.h"
#include "wrappers.h"

//prototype for the kernel
__global__ void d_crack_kernel(unsigned char * hash, int hashLen,
                                int length, unsigned char * d_result,
                                int d_result_size);

__global__ void d_generate_kernel(unsigned char * passwords, int length, unsigned long n,
                                    unsigned char * d_result);

//__device__ int d_powerOf(int val, int size);

//constant array containing all the possible characters in the password

/*malloccmp
* a compare like function that compares two strings of length.  It simply
* compares the elements at each location.
*
* @params:
*   str1   - an unsigned char pointer to the first character in string 1
*   str2   - an unsigned char pointer to the first character in string 2
*   length - the length of str1 and str2, the number of items compared.
*/
int malloccmp(unsigned char * str1, unsigned char * str2, int length) {
  for (int i = 0; i < length; i++) {
    if (str1[i] != str2[i]) {
      return 0;
    }
  }
  return 1;
}


/*d_crack
*
* Sets up and calls the kernal to brute-force a password hash.
*
* @params
*   hash    - the password hash to brute-force
*   hashLen - the length of the hash
*   outpass - the result password to return
*/
float d_crack(unsigned char * hash, int hashLen, unsigned char * outpass) {

    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use cuda functions to do the timing
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));

    unsigned long size = 2 * NUMCHARS * sizeof(unsigned char);
    unsigned long outsize = pow(NUMCHARS, 2) * 3;

    unsigned char * d_passwords;
    CHECK(cudaMalloc((void**)&d_passwords, size));
    unsigned char * d_result;
    CHECK(cudaMalloc((void**)&d_result, outsize));


    //Copy the starting passwords array and valid characters to the GPU
    CHECK(cudaMemcpyToSymbol(VALID_CHARS, VALID_CHARS_CPU, NUMCHARS * sizeof(char)));
    CHECK(cudaMemcpy(d_passwords, STARTING_PASSES, 2 * NUMCHARS, cudaMemcpyHostToDevice));


    // Beginning of Four-way Radix Sort
    // We need a block size of 256
    dim3 block(BLOCKDIM, 1, 1);
    dim3 grid(1, 1, 1);

    d_generate_kernel<<<grid, block>>>(d_passwords, 1, NUMCHARS, d_result);

    CHECK(cudaDeviceSynchronize());

    unsigned char * passwords = (unsigned char *) Malloc(outsize);
    CHECK(cudaMemcpy(passwords, d_result, outsize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_passwords));
    CHECK(cudaFree(d_result));

    free(passwords);

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
__global__ void d_generate_kernel(unsigned char * passwords, int length, unsigned long n,
                                    unsigned char * d_result) {
  unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    unsigned long r_index = index * (length + 2) * NUMCHARS;
    unsigned long p_index = index * (length + 1);
  }
}
