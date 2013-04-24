/**
 * @file
 *
 * @brief a wrapper around a memory block on the device for automatic removal
 *  at the end of the scope.
 */
#ifndef CUDAMEMBLOCK_HPP
#define CUDAMEMBLOCK_HPP
#include <cuda.h>
#include <helper_cuda.h>

/* the data is public so that we can easily give it to cuda to work
   with. While this is a potential risk, the programmer is trusted as
   to not abuse the data making it invalid. */
template<class T>
class cudaMemBlock {
public:
  // mem is the memory on the device,
  // host is the memory on the host (usually the RAM memory)
  T *mem, *host;
  // length of the array in number of elements.
  unsigned int size;
  /* if the user has defined the host, the user is responsible for
     cleaning it up, not us */
  bool userDefinedHost;

  /* Size: size in number of elements (NOT bytes) */
  cudaMemBlock(unsigned int Size);
  /**
   * Size: size in number of elements (NOT bytes)
   * Host: the memory on the host-device. The caller remains
   *       responsible for clearing this memory.
   */
  cudaMemBlock(unsigned int Size, T *Host);
  

  virtual ~cudaMemBlock();
  /**
   * @brief Size: the size in number of elements, not in number of
   * bytes!
   * Kind: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, ...
   */
  void memcpy(int Size, enum cudaMemcpyKind Kind);
  /* this will copy the whole block */
  void memcpy(enum cudaMemcpyKind);

  T& operator[](int n);
  const T& operator[](int n) const;





private:
  // make non-copyable
  cudaMemBlock( const cudaMemBlock&);
  const cudaMemBlock& operator=( const cudaMemBlock& );
};

template<class T>
cudaMemBlock<T>::cudaMemBlock(unsigned int Size):
  mem(0), host(0), size(Size), userDefinedHost(false) {
  checkCudaErrors(cudaMalloc( (void**) &mem, size*sizeof(T) ));
  //host = new T[size];
  host = (T*)malloc(size*sizeof(T));
  //host = (T *) calloc( size, sizeof(T) );
  memset(host,0,size);
}

template<class T>
cudaMemBlock<T>::cudaMemBlock(unsigned int Size, T *Host):
  mem(0), host(Host), size(Size), userDefinedHost(true) {
  checkCudaErrors(cudaMalloc( (void**) &mem, size*sizeof(T) ));
}

template<class T>
cudaMemBlock<T>::~cudaMemBlock() {
  getLastCudaError("we made a mistake earlier!");
  checkCudaErrors(cudaFree(mem));
  // delete host and set it to 0
  if (!userDefinedHost) {
    free(host);
    host = 0;
  }
}

template<class T>
void cudaMemBlock<T>::memcpy(enum cudaMemcpyKind Kind) {
  this -> memcpy(size, Kind);
}

template<class T>
void cudaMemBlock<T>::memcpy(int Size, enum cudaMemcpyKind Kind) {
  getLastCudaError("we made a mistake earlier!");
  checkCudaErrors(cudaMemcpy( host, mem, Size * sizeof(T), Kind));
}

template<class T>
T& cudaMemBlock<T>::operator[](int n) {
  return host[n];
}
template<class T>
const T& cudaMemBlock<T>::operator[](int n) const {
  return host[n];
}
#endif // CUDAMEMBLOCK_HPP
