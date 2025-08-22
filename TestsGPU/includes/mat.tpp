#include "mat.h"
#include <random>
#include <type_traits>
#include <vector>



template<typename T>
T& Mat<T>::operator()(int r, int c){
    return data[r * cols + c];
}

template<typename T>
const T& Mat<T>::operator()(int r, int c) const{
    return data[r * cols + c];
}

template<typename T>
Mat<T>::Mat(const Mat<T> &copiedMat) {
        data = copiedMat.data;
}


template<typename T>
void Mat<T>::fillRandNumber(const T downRange, const T UpRange) {
 
    if constexpr (std::is_same<T,int>::value) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distrib(downRange, UpRange);

        for(int i = 0; i < rows*cols; i++){
            data[i] = distrib(gen);
        }
    }

    if constexpr (std::is_same<T,float>::value){

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(downRange, UpRange);

        for(int i = 0; i < rows*cols; i++){
            data[i] = distrib(gen);
        }
    }
}

