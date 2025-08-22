#pragma once
#include <vector>

template<typename T>
class optMat {

using optMatrix = std::vector<T>;

private:
    optMatrix optData;
    int dim;

public:
    optMat(int taille, T init = T())
    : optData(taille * taille, init), dim(taille) {};

    optMat(const optMat<T> &copiedMat);

    int size() const ;
    T& operator()(int i, int j);
    const T& operator()(int i, int j) const;

    optMat<T> operator+(const optMat<T> &A);
    optMat<T> operator*(const optMat<T> &A);


    optMat<T> hadamardProduct(const optMat<T> &A);
    void fillRandNumber(const T downRange, const T UpRange);

};

#include "optimisedMat.tpp"