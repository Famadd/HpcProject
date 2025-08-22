#include "optimisedMat.h"


void optimisedMatricialProductTime(int taille, auto downRange, auto UpRange) {
        
        optMat<int> A(taille);
        optMat<int> B(taille);
        optMat<int> C(taille);
        A.fillRandNumber(downRange,UpRange);
        B.fillRandNumber(downRange,UpRange);

        const auto start{std::chrono::steady_clock::now()};
        C = A*B;
        const auto finish{std::chrono::steady_clock::now()};
        
        const std::chrono::duration<double> elapsed_seconds{finish - start};
        std::cout << "Matricial Product time : " << elapsed_seconds << std::endl;
}

void optimisedMatricialAdditionTime(int taille, auto downRange, auto UpRange) {
        optMat<int> A(taille);
        optMat<int> B(taille);
        optMat<int> C(taille);
        A.fillRandNumber(downRange,UpRange);
        B.fillRandNumber(downRange,UpRange);

        const auto start{std::chrono::steady_clock::now()};
        C = A+B;
        const auto finish{std::chrono::steady_clock::now()};
        
        const std::chrono::duration<double> elapsed_seconds{finish - start};
        std::cout << "Matricial Addition time : " << elapsed_seconds << std::endl;
}

void optimisedMatricialHadamardProductTime(int taille, auto downRange, auto UpRange) {
        optMat<int> A(taille);
        optMat<int> B(taille);
        optMat<int> C(taille);
        A.fillRandNumber(downRange,UpRange);
        B.fillRandNumber(downRange,UpRange);

        const auto start{std::chrono::steady_clock::now()};
        C = A.hadamardProduct(B);
        const auto finish{std::chrono::steady_clock::now()};
        
        const std::chrono::duration<double> elapsed_seconds{finish - start};
        std::cout << "Matricial Hadamard Product time : " << elapsed_seconds << std::endl;
}