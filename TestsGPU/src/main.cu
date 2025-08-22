#include "../includes/config.h"
#include "mat.h"

int main(int argc, char **argv) {

    int taille = 0;

    std::cout << "Choissisez la taille (nombres de colonnes ou lignes) des matrices à calculer" << std::endl;
    do{
        std::cout << "La taille doit etre comprise entre 0 et 15000" << std::endl;
        std::cin >> taille;
    } while (taille < 0 || taille > 15000); 

    size_t size = taille*taille*sizeof(float);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((taille+15)/16, (taille+15)/16);


    Mat<float> h_A(taille,taille,0.0f);
    Mat<float> h_B(taille,taille,0.0f);
    Mat<float> h_C(taille,taille,0.0f);

    h_A.fillRandNumber(0,2000.0f);
    h_B.fillRandNumber(0,2000.0f);
    h_C.fillRandNumber(0,2000.0f);

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(&d_A, h_A.getDataPtr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(&d_B, h_B.getDataPtr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(&d_C, h_C.getDataPtr(), size, cudaMemcpyHostToDevice);

    matrixProductTimeCalculus<float>(h_A, h_B, h_C,taille);


    matrixAdditionTimeCalculus<float>(h_A, h_B, h_C,taille);


    hadamardProductTimeCalculus(numBlocks, threadsPerBlock, d_A, d_B, d_C, taille, taille);

    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
