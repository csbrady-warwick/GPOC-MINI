#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstring>
//CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __CUDACC__
template<typename T=double, typename T2=T>
__global__ void setArray(T* data, int64_t size, T2 value){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size){
        data[idx] = value;
    }
}
#endif

#endif