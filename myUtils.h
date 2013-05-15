#ifndef _MYUTILS_H_
#define _MYUTILS_H_
#include <iostream>
#include <fstream>
//#define oprintf(...) mexPrintf(__VA_ARGS__)
#define oprintf(...) printf(__VA_ARGS__)

typedef unsigned int uint;

// output for debug purposes, always to the command line
#define dout std::stderr

#define myCalloc(name, type, size)           \
  name = (type) calloc(size, sizeof(type) ); \
  if (!name)                                 \
    printf("Memory allocation failure\n")

/**
 * write out the array to a stream.
 * 
 */
template<class T>
void writeArray(const T *arr, int size, std::ostream &out, char delimiter = '\n') {
  for (int i = 0; i < size; ++i) {
    if (arr[i] != -1)
      out  << arr[i] << delimiter;
  }
}

/**
 * write out the data from an array.
 * size is in number of elements
 */
template<class T>
void writeArray(const T *arr, int size, const char *filename, char delimiter = '\n') {
  std::ofstream out(filename);
  writeArray(arr, size, out, delimiter);
}


/**
 * calculate the total amount of elements in the given dimensions
 */
static int calcDimSize(int dims, const int *dimSizes)
{
  int total = 1;
  if (dims == 0) 
    return 0;
  for (int i = 0; i < dims; ++i) {
    total *= dimSizes[i];
  }
  return total;
}


/**
 * speaks for itself
 */
static bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

// copied from the NVidia reduction example
static unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
/**
 * calculate the bandwidth used.
 */
static float bandwidthInMBs(int memSize, int transactions, float elapsedTimeInMs) {
  float bw = (1.e3f * memSize * (float)transactions) /
             (elapsedTimeInMs * (float)(1 << 20));
  return bw;
}
#endif /* _MYUTILS_H_ */
