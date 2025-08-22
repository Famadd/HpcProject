#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include "mat.h"

template <typename T>
void matrixProductTimeCalculus(const Mat<T> &h_A, const Mat<T> &h_B, Mat<T> &h_C, int taille) {
    
    size_t size = taille * taille * sizeof(T);

    Mat<float> A(taille * taille,0.0f);
    Mat<float> B(taille * taille,0.0f);
    Mat<float> C(taille * taille,0.0f);

    float* d_A = A.getDataPtr();
    float* d_B = B.getDataPtr();
    float* d_C = C.getDataPtr();

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A.getDataPtr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.getDataPtr(), size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        taille, taille, taille,
        &alpha,
        d_B, taille,
        d_A, taille,
        &beta,
        d_C, taille);

    cudaMemcpy((h_C.getDataPtr()), d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    

    std::cout << "Temps d'execution GPU - Produit matriciel : " << ms / 1000.0 << " s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

template<typename T>
void matrixAdditionTimeCalculus(const Mat<T> &h_A, const Mat<T> &h_B, Mat<T> &h_C, int taille) {
    
    size_t size = taille * taille * sizeof(T);

    Mat<float> A(taille * taille,0.0f);
    Mat<float> B(taille * taille,0.0f);
    Mat<float> C(taille * taille,0.0f);

    float* d_A = A.getDataPtr();
    float* d_B = B.getDataPtr();
    float* d_C = C.getDataPtr();

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A.getDataPtr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.getDataPtr(), size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cublasSgeam(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N, 
    taille, taille,
    &alpha,
    d_A, taille,
    &beta,
    d_B, taille,
    d_C, taille
    );

    cudaMemcpy((h_C.getDataPtr()), d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    

    std::cout << "Temps d'execution GPU - Addition matriciel : " << ms / 1000.0 << " s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

template<typename T>
__global__ void hadamardProductKernel(const T*d_A, const T*d_B, T* d_C, int rows, int cols) {
        const int TILE_SIZE = 16;
        
        // Indices locaux et globaux
        int localRow = threadIdx.y;
        int localCol = threadIdx.x;
        int globalRow = blockIdx.y * TILE_SIZE + localRow;
        int globalCol = blockIdx.x * TILE_SIZE + localCol;

        // Mémoire partagée pour le tile
        __shared__ T tileA[TILE_SIZE][TILE_SIZE];
        __shared__ T tileB[TILE_SIZE][TILE_SIZE];

        // Charger les éléments du tile dans la mémoire partagée
        if(globalRow < rows && globalCol < cols) {
            int idx = globalRow * cols + globalCol;
            tileA[localRow][localCol] = d_A[idx];
            tileB[localRow][localCol] = d_B[idx];
        }
        __syncthreads(); // Synchronisation avant le calcul

        // Calcul Hadamard sur le tile
        if(globalRow < rows && globalCol < cols) {
            int idx = globalRow * cols + globalCol;
            d_C[idx] = tileA[localRow][localCol] * tileB[localRow][localCol];
    }
}


template<typename T>
void hadamardProductTimeCalculus(dim3 numblocks, dim3 threadPerblocks,const T*d_A, const T*d_B, T* d_C, int rows, int cols){

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((cols+15)/16, (rows+15)/16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    hadamardProductKernel<<<numBlocks,threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    

    std::cout << "Temps d'execution GPU - Produit de Hadamard : " << ms / 1000.0 << " s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

};