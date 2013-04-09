/* a wrapper around a memory block on the device for automatic removal
   at the end of the scope. */
#ifndef CUDAMEMBLOCK_HPP
#define CUDAMEMBLOCK_HPP
#include "cuda.h"
#include "helper_cuda.h"

/* the data is public so that we can easily give it to cuda to work
   with. While this is a potential risk, the programmer is trusted as
   to not abuse the data making it invalid. */
template<class T>
class cudaMemBlock {
public:
  T *mem, *host;

  unsigned int size;
  /* if the user has defined the host, the user is responsible for
     cleaning it up, not us */
  bool userDefinedHost;
  cudaMemBlock(unsigned int Size):
    mem(0), host(0), size(Size), userDefinedHost(false) {
    checkCudaErrors(cudaMalloc( (void**) &mem, size*sizeof(T) ));
    //host = new T[size];
    host = (T*)malloc(size*sizeof(T));
    //host = (T *) calloc( size, sizeof(T) );
    memset(host,0,size);
  }
  cudaMemBlock(unsigned int Size, T *Host):
      mem(0), host(Host), size(Size),  userDefinedHost(true) {
    checkCudaErrors(cudaMalloc( (void**) &mem, size*sizeof(T) ));
  }

  virtual ~cudaMemBlock() {
    getLastCudaError("we made a mistake earlier!");
    checkCudaErrors(cudaFree(mem));
    if (!userDefinedHost) {
      //delete host;
      free(host);
      host = 0;
    }
  }

  void memcpy(enum cudaMemcpyKind Kind) {
    this->memcpy(size, Kind);
  }
  /* Size: the size in number of elements, not in number of bytes! */
  void memcpy(int Size, enum cudaMemcpyKind Kind) {
    getLastCudaError("we made a mistake earlier!");
    checkCudaErrors(cudaMemcpy( host, mem, Size * sizeof(T), Kind));
  }

  T& operator[](int n) {
    return host[n];
  }
  const T& operator[](int n) const {
    return host[n];
  }

private:
  // make non-copyable
  cudaMemBlock( const cudaMemBlock&);
  const cudaMemBlock& operator=( const cudaMemBlock& );
};
#endif // CUDAMEMBLOCK_HPP
