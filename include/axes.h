#ifndef AXES_H
#define AXES_H

#ifndef TYPE
#pragma warning "TYPE not defined, defaulting to double"
#define TYPE TYPE
#endif

template<typename T, int rank>
class deviceAxis;

template<typename T=TYPE, int rank=2, bool host=false>
class axis;

namespace {
    template<typename T, int i_rank>
    __global__ void initDeviceAxis(deviceAxis<T,i_rank> *axis, int64_t *dims_centre, 
                                      int64_t *dims_edge, T *data_centre, T *data_edge, T*delta, int64_t nghosts){
        new(axis) deviceAxis<T,i_rank>(dims_centre, dims_edge, data_centre, data_edge, delta, nghosts);
    }
};

template<typename T, int i_rank>
__global__ void printAxis(deviceAxis<T,i_rank> *axis){
    for (int i=0;i<axis->rank;i++){
        printf("Axis: %d\n", i);
        printf("Centre\n");
        printf("Count: %ld\n", axis->dims_centre[i]);
        printf("Offset: %ld\n", axis->offset_centre[i]);
        printf("Ghosts: %ld\n", axis->nghosts);
        for (int j=1-axis->nghosts;j<1+axis->dims_centre[i];j++){
            printf("%f ", axis->getCentre(i+1,j));
        }
        printf("\n\n");
    }
    printf("\n");
}

template<typename T, int i_rank>
__global__ void printAxisEdge(deviceAxis<T,i_rank> *axis){
    for (int i=0;i<axis->rank;i++){
        printf("Axis: %d\n", i);
        printf("Edge\n");
        printf("Count: %ld\n", axis->dims_edge[i]);
        printf("Offset: %ld\n", axis->offset_edge[i]);
        for (int j=0;j<axis->dims_edge[i];j++){
            printf("%f ", axis->data_edge[axis->offset_edge[i]+j]);
        }
        printf("\n\n");
    }
    printf("\n");
}

template<typename T, int i_rank, bool host>
class axis{
    private:
    template<int level=0, typename current, typename... Args>
    void set(current first, Args... args){
        if constexpr(level==0){
            size_centre = 0;
            size_edge = 0;
        }
        dims_centre[level] = first + 2*nghosts;
        dims_edge[level] = first + 2*nghosts + 1;
        size_centre += dims_centre[level];
        size_edge += dims_edge[level];
        if constexpr(sizeof...(args)>0){
            set<level+1>(args...);
        }
    }

    void cleanHost(){
        if (data_centre) delete[](data_centre);
        data_centre=nullptr;

        if (data_edge) delete[](data_edge);
        data_edge=nullptr;
    }

    void cleanDevice(){
        if constexpr(host){
            //Nothing to do on host
        } else {
            if (d_dims_centre) cudaFree(d_dims_centre);
            if (d_dims_edge) cudaFree(d_dims_edge);
            if (d_data_centre) cudaFree(d_data_centre);
            if (d_data_edge) cudaFree(d_data_edge);
            if (d_axis) cudaFree(d_axis);
            if (d_delta) cudaFree(d_delta);
            d_dims_centre = nullptr;
            d_dims_edge = nullptr;
            d_data_centre = nullptr;
            d_data_edge = nullptr;
            d_axis = nullptr;
            d_delta = nullptr;
        }
    }
    public:
    using type = T;
    static constexpr int rank=i_rank;
    //host data
    int64_t dims_centre[rank];
    int64_t dims_edge[rank];
    int64_t offset_centre[rank];
    int64_t offset_edge[rank];
    int64_t nghosts;
    int64_t size_centre;
    int64_t size_edge;
    T* data_centre=nullptr;
    T* data_edge=nullptr;
    T delta[rank];

    //device data
    int64_t *d_dims_centre=nullptr;
    int64_t *d_dims_edge=nullptr;
    T *d_data_centre=nullptr;
    T *d_data_edge=nullptr;
    deviceAxis<T,rank> *d_axis=nullptr;
    T* d_delta=nullptr;

    template<typename... Args>
    axis(int64_t nghosts, Args... args){
        if constexpr(sizeof...(args) == rank-1){
            this->nghosts = 0;
            set(nghosts, args...);
        } else {
            this->nghosts = nghosts;
            set(args...);
        }
        offset_centre[0] = 0;
        offset_edge[0] = 0;
        for (int i=1;i<rank;i++){
            offset_centre[i] = offset_centre[i-1] + dims_centre[i-1];
            offset_edge[i] = offset_edge[i-1] + dims_edge[i-1];
        }
        cudaMalloc(&d_dims_centre, sizeof(int64_t) * rank);
        cudaMalloc(&d_dims_edge, sizeof(int64_t) * rank);
        cudaMemcpy(d_dims_centre, dims_centre, sizeof(int64_t) * rank, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dims_edge, dims_edge, sizeof(int64_t) * rank, cudaMemcpyHostToDevice);
        cudaMalloc(&d_data_centre, sizeof(T) * size_centre);
        cudaMalloc(&d_data_edge, sizeof(T) * size_edge);
        cudaMalloc(&d_axis, sizeof(deviceAxis<T,rank>));
        cudaMalloc(&d_delta, sizeof(T)*rank);
        initDeviceAxis<<<1,1>>>(d_axis, d_dims_centre, d_dims_edge, d_data_centre, d_data_edge, d_delta, nghosts);
        setArray<<<1,1>>>(d_data_centre,size_centre,0.0);
        setArray<<<1,1>>>(d_data_edge,size_edge,0.0);
        data_centre = new T[size_centre];
        data_edge = new T[size_edge];
    }

    void buildAxis(int direction, T LB, T UB){
        int64_t bindex = offset_centre[direction-1];
        T delta = T(UB-LB)/T(dims_centre[direction-1]-2*nghosts);
        this->delta[direction-1] = delta;
        T TLB = LB - nghosts*delta + delta/2.0;
        data_centre[bindex] = TLB;
        for (int64_t i = bindex+1; i < bindex + dims_centre[direction-1]; i++){
            data_centre[i] = data_centre[i-1] + delta;
        }
        if constexpr(!host){
            cudaMemcpy(d_data_centre + bindex, data_centre + bindex, sizeof(T)*dims_centre[direction-1], cudaMemcpyHostToDevice);
            cudaMemcpy(d_delta + direction-1, &delta, sizeof(T), cudaMemcpyHostToDevice);
        }

        bindex = offset_edge[direction-1];
        TLB = LB - nghosts*delta;
        data_edge[bindex] = TLB;
        for (int64_t i = bindex+1; i < bindex + dims_edge[direction-1]; i++){
            data_edge[i] = data_edge[i-1] + delta;
        }
        if constexpr(!host){
            cudaMemcpy(d_data_edge + bindex, data_edge + bindex, sizeof(T)*dims_edge[direction-1], cudaMemcpyHostToDevice);
        }
    }

    void buildX(T LB, T UB){
        buildAxis(1, LB, UB);
    }

    void buildY(T LB, T UB){
        static_assert(rank>1, "Cannot build Y axis in 1D");
        buildAxis(2, LB, UB);
    }

    void buildZ(T LB, T UB){
        static_assert(rank>2, "Cannot build Z axis in 1D or 2D");
        buildAxis(3, LB, UB);
    }

    deviceAxis<T,rank>* onDevice(){
        return d_axis;
    }

    //Casting operator to deviceAxis
    operator deviceAxis<T,rank>*(){
        return d_axis;
    }

    T* getCentre(){
        return data_centre;
    }

    T* getEdge(){
        return data_edge;
    }

    void putCentre(T* data){
        if (data!=data_centre) memcpy(data_centre, data, sizeof(T)*size_centre);
        if constexpr(!host){
            cudaMemcpy(d_data_centre, data_centre, sizeof(T)*size_centre, cudaMemcpyHostToDevice);
        }
    }

    void putEdge(T* data){
        if (data!=data_edge) memcpy(data_edge, data, sizeof(T)*size_edge);
        if constexpr(!host){
            cudaMemcpy(d_data_edge, data_edge, sizeof(T)*size_edge, cudaMemcpyHostToDevice);
        }
    }

    T& getCentre(int direction, size_t element){
        return data_centre[offset_centre[direction-1]+element-1 + nghosts];
    }

    T& getEdge(int direction, size_t element){
        return data_edge[offset_edge[direction-1]+element-1 + nghosts];
    }

    T& xc(size_t element){
        return data_centre[offset_centre[0]+element-1 + nghosts];
    }

    T& yc(size_t element){
        static_assert(rank>1, "Cannot access yc in 1D");
        return data_centre[offset_centre[1]+element-1 + nghosts];
    }

    T& zc(size_t element){
        static_assert(rank>2, "Cannot access zc in 1D or 2D");
        return data_centre[offset_centre[2]+element-1 + nghosts];
    }

    T&xb(size_t element){
        return data_edge[offset_edge[0]+element-1 + nghosts];
    }

    T& yb(size_t element){
        static_assert(rank>1, "Cannot access yb in 1D");
        return data_edge[offset_edge[1]+element-1 + nghosts];
    }

    void syncToDevice(){
        if constexpr(!host){
            cudaMemcpy(data_centre, d_data_centre, sizeof(T)*size_centre, cudaMemcpyDeviceToHost);
            cudaMemcpy(data_edge, d_data_edge, sizeof(T)*size_edge, cudaMemcpyDeviceToHost);
        }
    }

    int64_t getNCells(int direction){
        return dims_centre[direction-1];
    }

    int64_t getNEdges(int direction){
        return dims_edge[direction-1];
    }

    int64_t nx(){
        return dims_centre[0]-2*nghosts;
    }

    int64_t ny(){
        static_assert(rank>1, "Cannot access getNY in 1D");
        return dims_centre[1]-2*nghosts;
    }

    int64_t nz(){
        static_assert(rank>2, "Cannot access getNZ in 1D or 2D");
        return dims_centre[2]-2*nghosts;
    }

    T getDelta(int direction){
        return delta[direction-1];
    }

    T dx(){
        return delta[0];
    }

    T dy(){
        static_assert(rank>1, "Cannot access getDY in 1D");
        return delta[1];
    }

    T dz(){
        static_assert(rank>2, "Cannot access getDZ in 1D or 2D");
        return delta[2];
    }

    ~axis(){
        cleanHost();
        cleanDevice();
    }

};

template<typename T, int i_rank>
class deviceAxis{
    public:
    using type = T;
    static constexpr int rank=i_rank;
    int64_t *dims_centre;
    int64_t *dims_edge;
    T *data_centre;
    T *data_edge;
    int64_t size_centre;
    int64_t size_edge;
    int64_t offset_centre[rank];
    int64_t offset_edge[rank];
    int64_t nghosts;
    T *delta = nullptr;

    __device__ deviceAxis(int64_t *dims_centre, int64_t *dims_edge, T *data_centre, T *data_edge, T *delta, int64_t nghosts){
        this->dims_centre = dims_centre;
        this->dims_edge = dims_edge;

        this->data_centre = data_centre;
        this->data_edge = data_edge;

        this->delta = delta;

        this->nghosts = nghosts;
        this->size_centre = 1;
        this->size_edge = 1;
        this->offset_centre[0] = 0;
        this->offset_edge[0] = 0;
        for (int i=1;i<rank;i++){
            this->offset_centre[i] = this->offset_centre[i-1] + dims_centre[i-1];
            this->offset_edge[i] = this->offset_edge[i-1] + dims_edge[i-1];
        }
        for (int i=0;i<rank;i++){
            this->size_centre *= dims_centre[i];
            this->size_edge *= dims_edge[i];
        }

    }
    __device__ T& getCentre(int direction, size_t element){
        return data_centre[offset_centre[direction-1]+element-1 + nghosts];
    }

    __device__ T& getEdge(int direction, size_t element){
        return data_edge[offset_edge[direction-1]+element-1 + nghosts];
    }

    __device__ T& xc(size_t element){
        return data_centre[offset_centre[0]+element-1 + nghosts];
    }

    __device__ T& yc(size_t element){
        static_assert(rank>1, "Cannot access yc in 1D");
        return data_centre[offset_centre[1]+element-1 + nghosts];
    }

    __device__ T& zc(size_t element){
        static_assert(rank>2, "Cannot access zc in 1D or 2D");
        return data_centre[offset_centre[2]+element-1 + nghosts];
    }

    __device__ T&xb(size_t element){
        return data_edge[offset_edge[0]+element-1 + nghosts];
    }

    __device__ T& yb(size_t element){
        static_assert(rank>1, "Cannot access yb in 1D");
        return data_edge[offset_edge[1]+element-1 + nghosts];
    }

    __device__ int64_t getNCells(int direction){
        return dims_centre[direction-1];
    }

    __device__ int64_t getNEdges(int direction){
        return dims_edge[direction-1];
    }

    __device__ int64_t nx(){
        return dims_centre[0]-2*nghosts;
    }

    __device__ int64_t ny(){
        static_assert(rank>1, "Cannot access getNY in 1D");
        return dims_centre[1] - 2*nghosts;
    }

    __device__ int64_t nz(){
        static_assert(rank>2, "Cannot access getNZ in 1D or 2D");
        return dims_centre[2] - 2*nghosts;
    }

    __device__ T getDelta(int direction){
        return delta[direction-1];
    }

    __device__ T dx(){
        return delta[0];
    }

    __device__ T dy(){
        static_assert(rank>1, "Cannot access getDY in 1D");
        return delta[1];
    }

    __device__ T dz(){
        static_assert(rank>2, "Cannot access getDZ in 1D or 2D");
        return delta[2];
    }

};

#endif