#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <openssl/md5.h>
#include "wrappers.h"
#include "d_hybrid_sort.h"
#include "config.h"

void parseCommandArgs(int argc, char * argv[], int * lower_bound,
                      int * upper_bound, bool * quiet);
void printUsage();

/*main
*
* Processes command line arguments, and if successful it calls the d_hybrid_sort
* function to launch our CUDA kernal and sort a random input array.
* When successful, it will print out the time in miliseconds in input size.
*
* @params:
*   argc - # of arguments in argv
*   argv - command-line arguments
*
*   @authors: Gurney Buchanan <@gurnben>, Eric Cambel <@cambelem>, and adapted
*              from code and teaching by Dr. Cindy Norris <@cindyanorris>
*/
int main(int argc, char * argv[]) {
    int lower_bound = 0, upper_bound = 0;
    bool quiet = false;

    parseCommandArgs(argc, argv, &lower_bound, &upper_bound, &quiet);

    srand(1);

    for (int i = lower_bound; i <= upper_bound; i++) {
      unsigned int num_elems = (1 << i);
      unsigned int* h_in = new unsigned int[num_elems];
      unsigned int* h_in_rand = new unsigned int[num_elems];

      for (unsigned int j = 0; j < num_elems; j++) {
        h_in[j] = (num_elems - 1) - j;
        h_in_rand[j] = rand() % num_elems;
      }

      float time = 0;
      for (unsigned int j = 0; j < 2; j++) {
        time = d_sort(h_in_rand, num_elems, quiet);
      }

      printf("Hybrid GPU Sort took %f milliseconds to sort %e (2^%d) numbers.\n",
              time, pow(2, i), i);

      delete[] h_in;
      delete[] h_in_rand;
    }
    return EXIT_SUCCESS;
}

/*parseCommandArgs
*
* This function processes the command line arguments given to the program.
*
* the proper use is:
*   ./hybrid_sort <lower_bound> <upper_bound>
*
* @params:
*   argc        - the number of arguments in argv
*   argv        - the arguments to the utility
*   lower_bound - a pointer to a lower_bound variable
*   upper_bound - a pointer to an upper_bound variable
*/
void parseCommandArgs(int argc, char * argv[], int * lower_bound,
                      int * upper_bound, bool * quiet) {
    if (argc < 3) {
      printUsage();
      //exit because the input was incorrect
      exit(EXIT_FAILURE);
    }
    else {
      (*lower_bound) = atoi(argv[argc - 2]);
      (*upper_bound) = atoi(argv[argc - 1]);
    }
    if (std::string(argv[argc - 3]) == "-q") {
      (*quiet) = true;
    }
}

/*printUsage
*
* Prints the usage information for this application.
*/
void printUsage()
{
    printf("This application takes as input a lower and upper bound for\n");
    printf("data sizes.  The input lower and upper bounds are taken in as\n");
    printf("powers of 2.  For example, input 24 as a lower bound will sort\n");
    printf("2^24 elements.  Warning: inputs above 26 are very slow!\n");
    printf("\nusage: hybrid_sort <lower bound> <upper bound>\n");
    printf("\t<lower_bound> will be treated as a power of 2 and is inclusive\n");
    printf("\t<upper_bound> will be treated as a power of 2 and is inclusive\n");
    printf("Examples:\n");
    printf("\t./hybrid_sort 24 26\n");
}
