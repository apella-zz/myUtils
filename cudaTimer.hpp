#ifndef CUDATIMER_HPP
#define CUDATIMER_HPP
#include <cuda.h>
#include "helper_cuda.h"
class cudaTimer {
  cudaEvent_t _start, _stop;
  float elapsedTimeInMs;

  void createEvents() {
    checkCudaErrors(cudaEventCreate(&_start));
    checkCudaErrors(cudaEventCreate(&_stop));
  }
  void deleteEvents() {
    checkCudaErrors(cudaEventDestroy(_start));
    checkCudaErrors(cudaEventDestroy(_stop));
  }
  
public:
  cudaTimer(){
    createEvents();
  }
  
  void start() {
    checkCudaErrors(cudaEventRecord(_start, 0));
  }
  float stop() {
    checkCudaErrors(cudaEventRecord(_stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, _start, _stop));
    return elapsedTimeInMs;
  }
  void reset() {
    deleteEvents();
    createEvents();
  }

  float time() const {
    return elapsedTimeInMs;
  }
  
  virtual ~cudaTimer() {
    deleteEvents();
  }
  
};

#endif
