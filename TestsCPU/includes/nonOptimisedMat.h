#pragma once

#include <vector>

template<typename T>
class nonOptMat{

using nonOptMatrix = std::vector<std::vector<T>>;

private:
    nonOptMatrix nonOptdata;

public:
    nonOptMat(int taille, T init = T())
    : nonOptdata(taille, std::vector<T>(taille, init)) {};

    nonOptMat(const nonOptMat<T> &copiedMat);

    int size();
    T& operator()(int i, int j);

    const T& operator()(int i, int j) const;

    nonOptMat<T> operator+(const nonOptMat<T> &A);
    nonOptMat<T> operator*(const nonOptMat<T> &A);


    nonOptMat<T> hadamardProduct(const nonOptMat<T> &A);
    void fillRandNumber(const T downRange, const T UpRange);

};

#include "nonOptimisedMat.tpp"