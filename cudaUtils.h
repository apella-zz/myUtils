#ifndef _CUDAUTILS_H_
#define _CUDAUTILS_H_
#include <iostream>
#include <vector>
#include <cuda.h>

#include "cudaMemBlock.hpp"

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

#define SAFEWRITE(array, index, maxsize, value)                 \
  do {                                                          \
      if (index < maxsize)                                      \
        array[index] = value;                                   \
      else                                                      \
        printf("%s: index %d was out of bounds (max %d\n",      \
               __FUNCTION__, index, maxsize);                   \
  } while (false)
  

static void vectorTodim3(const std::vector<int> &v,
                  dim3 &d) {
  if (v.size() < 3)
    return;
  d.x = v[0];
  d.y = v[1];
  d.z = v[2];
}

/* copied from the Cuda code samples */
bool checkCUDAProfile(int dev, int min_runtime, int min_compute) {
  int runtimeVersion = 0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
  cudaRuntimeGetVersion(&runtimeVersion);
  fprintf(stderr,"  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
  fprintf(stderr,"  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

  if (runtimeVersion >= min_runtime && ((deviceProp.major<<4) + deviceProp.minor) >= min_compute) {
    return true;
  }
  else {
    return false;
  }
}

static void displayDeviceInfo() {
  cudaDeviceProp prop;
  int iDev;

  cudaGetDeviceCount(&iDev);
  if ((unsigned int)iDev == 0) {
    std::cout << "There is no CUDA device found!";
    return;
  }
  else {
    printf("There are %d CUDA devices in your computer. \n", iDev);
    for(int ii = 0; ii < iDev; ii ++){
      cudaGetDeviceProperties(&prop, ii);
      printf("------ General Information for CUDA device %d ------ \n", ii);
      printf("Name:  %s \n", prop.name);
      printf("Multiprocessor count:  %d \n", prop.multiProcessorCount);
      //printf("Total global memory: %ld \n", prop.totalGlobalMem);
      printf("Total global memory: %d \n", prop.totalGlobalMem);
      printf("---------------------------------------------------- \n\n");
    }
  }
  
}

static void chooseLatestGPU(bool verbose=false) {
  int iDev, dev;
  cudaGetDeviceCount(&iDev);
  if (iDev > 1) {
    int maxMajor = 0, maxMinor = 0, maxDev = 0;
    for (dev = 0; dev < iDev; dev++) {
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, dev);
      if (maxMajor > properties.major) {
        maxMajor = properties.major;
        maxMinor = properties.minor; // reset the minor count
        maxDev = dev;
      }
      else if (maxMajor == properties.major) {
        if (maxMinor > properties.minor) {
          maxMinor = properties.minor;
          maxDev = dev;
        }
      }
    }
    cudaSetDevice(maxDev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, maxDev);
    if (verbose) {
      std::cout << "chosen gpu: " << p.name << '\n';}
  }
}
static void quickCopy(cudaMemBlock<int>& block, const char *filename) {
  copyBackAndWrite(block.host, block.mem, block.size, filename);
}
static void quickCopy(cudaMemBlock<int>& block, int size, const char *filename) {
  copyBackAndWrite(block.host, block.mem, size, filename);
}
#endif /* _CUDAUTILS_H_ */
