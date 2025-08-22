#pragma once

template<typename T>
class Mat {

private:
    int rows, cols;
    T* data;

public:
    Mat(int r, int c, T init = T{})
    : rows(r), cols(c){
        size_t size = r * c * sizeof(T);
        if(cudaMallocManaged(&data,size)!= cudaSuccess){
            throw std::bad_alloc();
        }

        for(int i = 0; i < r*c; i++) data[i] = init;
    }

    Mat(const Mat<T> &copiedMat);

    T* getDataPtr() {
        return data;
    }

    const T* getDataPtr() const {
        return data;
    }

    __host__ __device__
    int size() const {
        return rows*cols;
    }

    __host__ __device__
    int nrows() const {return rows;}

    __host__ __device__
    int ncols() const {return cols;}

    __host__ __device__
    T& operator()(int r, int c);

    __host__ __device__
    const T& operator()(int r, int c) const;


    void fillRandNumber(const T downRange, const T UpRange);
};

#include "mat.tpp"