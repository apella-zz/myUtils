#ifndef _CUDAUTILS_H_
#define _CUDAUTILS_H_
#include <iostream>
#include <vector>
#include <cuda.h>

#include "cudaMemBlock.hpp"

static void vectorTodim3(const std::vector<int> &v,
                  dim3 &d) {
  if (v.size() < 3)
    return;
  d.x = v[0];
  d.y = v[1];
  d.z = v[2];
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

static void chooseLatestGPU() {
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
    std::cout << "chosen gpu: " << p.name << '\n';
  }
}
static void quickCopy(cudaMemBlock<int>& block, const char *filename) {
  copyBackAndWrite(block.host, block.mem, block.size, filename);
}
static void quickCopy(cudaMemBlock<int>& block, int size, const char *filename) {
  copyBackAndWrite(block.host, block.mem, size, filename);
}
#endif /* _CUDAUTILS_H_ */
