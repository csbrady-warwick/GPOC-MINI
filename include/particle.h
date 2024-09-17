#ifndef PARTICLE_H
#define PARTICLE_H

#ifndef TYPE
#pragma warning "TYPE not defined, defaulting to double"
#define TYPE double
#endif

#include <iostream>
#include <cstring>
//CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

#define PHYSICAL_DIMS 3

template<typename T=TYPE, int rank=2>
struct particle{
    T x[rank];
#ifndef USE_MOMENTUM
    T v[PHYSICAL_DIMS];
#else
    T p[PHYSICAL_DIMS];
#endif
    T weight=0.0;
};

template<typename T=TYPE, int rank=2>
class deviceParticleSet;

template<typename T=TYPE, int rank=2>
void* addParticles(T* data, int64_t npart);

template<typename T=TYPE, int dim=2>
class particleSet{

    public:
    using type = particle<T,dim>;
    const static int rank=dim;
    //Host data
    particle<T,rank> *data=nullptr;
    int64_t size;

    int64_t coffset=0;
    int64_t csize=0;

    //Device data
    particle<T,rank> *d_data=nullptr;
    deviceParticleSet<T,rank> *d_particleSet=nullptr;

    particleSet(int64_t size){
        this->size = size;
        cudaMalloc(&d_data, size*sizeof(particle<T,rank>));
        std::cout << "Allocating " << size << " particles" << std::endl;
        d_particleSet = addParticles(d_data, size);
    }

    ~particleSet(){
        if (data) delete[](data);
        if (d_data) cudaFree(d_data);
        if (d_particleSet) cudaFree(d_particleSet);
    }

    particle<T,rank>* get(){
        if (!data){
            data = new particle<T,rank>[size];
            cudaMemcpy(data, d_data, size*sizeof(particle<T,rank>), cudaMemcpyDeviceToHost);
        }
        return data;
    }

    void release(){
        if (data) delete[](data);
        data = nullptr;
    }

    void put(particleSet *p){
        cudaMemcpy(d_data, p->d_data, size*sizeof(particle<T,rank>), cudaMemcpyDeviceToDevice);
    }

    void put(particle<T,rank>* data, size_t size){
        cudaMemcpy(d_data, data, size*sizeof(particle<T,rank>), cudaMemcpyHostToDevice);
    }

    deviceParticleSet<T,rank>* onDevice(){
        return d_particleSet;
    }

};

template<typename T, int dim>
class deviceParticleSet{

    public:
    using type = particle<T,dim>;
    const static int rank=dim;

    particle<T,rank> *data=nullptr;
    T mass = 1.0/1862.0;
    T charge = -1.7e11*mass;

    int64_t size;

    __device__ deviceParticleSet(int64_t size, particle<T,rank> *data){
        this->size = size;
        this->data = data;
        memset(data, 0, size*sizeof(particle<T,dim>));
    }

    __device__ particle<T,dim>& operator()(int64_t idx){
        return data[idx-1];
    }

};

template<typename T, int rank>
__global__ void addParticlesInner(deviceParticleSet<T,rank> * ptr, int64_t size, particle<T,rank> * data){
    new(ptr) deviceParticleSet<T,rank>(size, data);
}

template<typename T, int rank>
 deviceParticleSet<T,rank> *addParticles(particle<T,rank>* data, int64_t npart){
    deviceParticleSet<T,rank> * ddata;
    cudaMalloc(&ddata, sizeof(deviceParticleSet<T,rank>));
    addParticlesInner<T,rank><<<1,1>>>(ddata, npart, data);
    return ddata;
}

#endif