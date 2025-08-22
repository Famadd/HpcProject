#include "nonOptimisedMat.h"


void nonOptimisedMatricialProductTime(int taille, auto downRange, auto UpRange) {
        
        nonOptMat<int> A(taille);
        nonOptMat<int> B(taille);
        nonOptMat<int> C(taille);
        A.fillRandNumber(downRange,UpRange);
        B.fillRandNumber(downRange,UpRange);

        const auto start{std::chrono::steady_clock::now()};
        C = A*B;
        const auto finish{std::chrono::steady_clock::now()};
        
        const std::chrono::duration<double> elapsed_seconds{finish - start};
        std::cout << "Matricial Product time : " << elapsed_seconds << std::endl;

        
}

void nonOptimisedMatricialAdditionTime(int taille, auto downRange, auto UpRange) {
        nonOptMat<int> A(taille);
        nonOptMat<int> B(taille);
        nonOptMat<int> C(taille);
        A.fillRandNumber(downRange,UpRange);
        B.fillRandNumber(downRange,UpRange);

        const auto start{std::chrono::steady_clock::now()};
        C = A+B;
        const auto finish{std::chrono::steady_clock::now()};
        
        const std::chrono::duration<double> elapsed_seconds{finish - start};
        std::cout << "Matricial Addition time : " << elapsed_seconds << std::endl;
}

void nonOptimisedMatricialHadamardProductTime(int taille, auto downRange, auto UpRange) {
        nonOptMat<int> A(taille);
        nonOptMat<int> B(taille);
        nonOptMat<int> C(taille);
        A.fillRandNumber(downRange,UpRange);
        B.fillRandNumber(downRange,UpRange);

        const auto start{std::chrono::steady_clock::now()};
        C = A.hadamardProduct(B);
        const auto finish{std::chrono::steady_clock::now()};
        
        const std::chrono::duration<double> elapsed_seconds{finish - start};
        std::cout << "Matricial Hadamard Product time : " << elapsed_seconds << std::endl;
}