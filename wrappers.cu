#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>

/* 
 * Malloc
 * wrapper for the malloc function
 * If malloc returns NULL then the memory allocation failed.
 *
*/

void * Malloc(size_t size)
{
   void * allocData = (void *) malloc(size);
   if (allocData == NULL)
   {
      printf("malloc failed: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
   }
   return allocData;
}
