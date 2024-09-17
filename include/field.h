#ifndef FIELD_H
#define FIELD_H

#ifndef TYPE
#pragma warning "TYPE not defined, defaulting to double"
#define TYPE double
#endif

#include <iostream>
#include <cstring>
//CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

template<typename T=TYPE, int dim=2>
class deviceField;

template<typename T=TYPE, int dim=2, bool host=false>
class field;

template<typename T, int rank>
deviceField<T,rank>* addField(field<T,rank> &f);

template<typename T, int dim, bool host>
class field{

    //friend deviceField<T,dim>* addField<T,dim>(field &f);
    public:
    //Host data
    int64_t size=0;
    int64_t nghosts;
    int64_t dims[dim];
    T* data=nullptr;
    using type = T;
    static constexpr int rank=dim;

    inline static int slot=0;

    //Device data
    int64_t *d_dims=nullptr;
    int64_t *d_nghosts=nullptr;
    T* d_data=nullptr;
    deviceField<T, dim> *d_field=nullptr;

    template<int level, typename current, typename... Args>
    void set(current first, Args... args){
        dims[level] = first + 2*nghosts;
        size *= dims[level];
        if constexpr(sizeof...(args)>0){
            set<level+1>(args...);
        }
    }

    template<int level=0, typename current, typename... Args>
    int64_t subscript(current first, Args... args){
        int64_t index=first + nghosts - 1;
        if constexpr(sizeof...(args)>0){
            //Fortran order, but I don't care
            index+=dims[level]*subscript<level+1>(args...);
        }
        return index;
    }

    void freeData(){
        if (data) delete[](data);
        data = nullptr;
    }

    void freeDeviceData(){
        if constexpr(host){
            //Nothing to do on host
        } else {
            #ifdef __CUDACC__
            if (d_data) cudaFree(d_data);
            d_data = nullptr;
            if (d_dims) cudaFree(d_dims);
            d_dims = nullptr;
            #endif
        }
    }

    void freeField(){
        if constexpr(host){
            //Nothing to do on host
        } else {
            if (d_field) cudaFree(d_field);
            d_field = nullptr;
        }
    }

    public:
    template<typename... Args>
    field(int64_t nghosts, Args... args){
        size = 1;
        static_assert(sizeof...(Args)==dim || sizeof...(Args)==dim-1, "Incorrect number of arguments");

        if constexpr(sizeof...(Args)==rank){
            this->nghosts = nghosts;
            set<0>(args...);
        } else {
            //Not actually specifying the number of ghosts
            this->nghosts = 0;
            set<0>(nghosts, args...);
        }
        if constexpr (!host){
#ifdef __CUDACC__
            cudaMalloc(&d_data, size*sizeof(T));
            cudaMalloc(&d_dims, rank*sizeof(int64_t));
            //Copy the dimensions to the device
            cudaMemcpy(d_dims, dims, rank*sizeof(int64_t), cudaMemcpyHostToDevice);
            setArray<<<(size+255)/256, 256>>>(d_data, size, 0);
            d_field = addField<T, dim>(*this);
#endif
        } else {
            data = new T[size];
            d_data = data;
        }
    }

    ~field(){
        freeDeviceData();
        freeData();
        freeField();
    }

    /**
     * @brief Get the data from the field
     * This gets a view onto the data, if the field is on the device, it will copy the data to the host
     * 
     */
    T* get(){
        if constexpr(!host){
            if (!data){
                data = (T*)malloc(size*sizeof(T));
            }
            cudaMemcpy(data, d_data, size*sizeof(T), cudaMemcpyDeviceToHost);
        }
        return data;
    }

    /**
     * @brief Release the data from the field
     * If the data is on the device then the host copy will be released
     */
    void releaseData(){
        if constexpr(host){
            //Noting to do on host
        }else{
            freeData();
        }
    }

    /**
     * @brief Set the data from another field
     * 
     * @param f 
     */
    void put(field *f){
        if constexpr(host){
            std::memcpy(data, f->data, size*sizeof(T));
        }else{
            cudaMemcpy(d_data, f->d_data, size*sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }

    void put(T* data){
        if constexpr(host){
            if (data!=this->data)
                std::memcpy(this->data, data, size*sizeof(T));
        }else{
            //Loop over data and get the max
            T max = -1e10;
            for (int i=0;i<size;i++){
                if (data[i]>max) max = data[i];
            }
            std::cout << "Max: " << max << std::endl;
            cudaMemcpy(d_data, data, size*sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    template<typename... Args>
    T& operator()(Args... args){
        T* d = get();
        return d[subscript(args...)];
    }

    void operator=(T value){
        T* d = get();
        for (int i=0; i<size; i++){
            d[i] = value;
        }
        put(d);
    }

    deviceField<T, dim>* onDevice(){
        return d_field;
    }

};

    template <typename T=TYPE, int dim=2>
    using hostField = field<T, dim, true>;

#ifdef __CUDACC__
template <typename T, int dim>
class deviceField{
    public:
    using type = T;
    template<int level=0, typename current, typename... Args>
    __device__ int64_t subscript(current first, Args... args){
        int64_t index=first + nghosts - 1;
        if constexpr(sizeof...(args)>0){
            //Fortran order, but I don't care
            index+=dims[level]*subscript<level+1>(args...);
        }
        return index;
    }
    public:
    static constexpr int rank=dim;
    static constexpr int sz = sizeof(T);
    int64_t size;
    int64_t nghosts;
    int64_t *dims;
    T* data=nullptr;

    __device__ deviceField(int64_t nghosts, int64_t* dims, T* data){
        this->dims = dims;
        this->data = data;
        this->nghosts = nghosts;
        size = 1;
        for (int i=0; i<dim; i++){
            size *= dims[i];
        }
    }

    template<typename... Args>
    __device__ T& operator()(Args... args){
        return data[subscript(args...)];
    }

    __device__ int64_t nx(){
        return dims[0]-2*nghosts;
    }

    __device__ int64_t ny(){
        static_assert(dim>1, "Field is not 2D");
        return dims[1]-2*nghosts;
    }

    __device__ int64_t nz(){
        static_assert(dim>2, "Field is not 3D");
        return dims[2]-2*nghosts;
    }
};

template<typename T, int rank>
__global__ void addFieldInner(deviceField<T,rank> * ptr, int64_t nghosts, int64_t* dims, T* data){
    new(ptr) deviceField<T,rank>(nghosts, dims, data);
}

template<typename T, int rank>
deviceField<T,rank>* addField(field<T,rank> &f){
    deviceField<T,rank> * dfield;
    cudaMalloc(&dfield, sizeof(deviceField<T,rank>));
    addFieldInner<T,rank><<<1,1>>>(dfield, f.nghosts, f.d_dims, f.d_data);
    return dfield;
}
#endif

template<typename T, int rank, typename T2>
void incFieldHost(field<T,rank,true> &f, T2 value=1){
    #pragma omp parallel for
    for (int i=0; i<f.size; i++){
        f.data[i] =i;
    }
}

#endif