#include <type_traits>
#include <chrono>
#include <random>
#include <iostream>
#include "nonOptimisedMat.h"

template<typename T>
using nonOptMatrix = std::vector<std::vector<T>>;

/**
 * @brief Constructeur par copie d'un objet Mat
 * 
 * @tparam T Type générique à choisir
 * @param copiedMat Matrice à copier
 */
template<typename T>
nonOptMat<T>::nonOptMat(const nonOptMat<T> &copiedMat) {
        nonOptdata = copiedMat.nonOptdata;
}
/**
 * @brief Surcharge de l'opérateur () pour l'accession d'une matrice en mode écriture
 * 
 * @tparam T Type générique à choisir
 * @param i Indice de ligne de matrice
 * @param j Indice de colonne de matrice
 * @return T& Retourne la valeur à l'indice d'une matrice donnée
 */
template<typename T>
T& nonOptMat<T>::operator()(int i, int j){
    return nonOptdata[i][j];
}

/**
 * @brief Surcharge de l'opérateur () pour l'accession d'une matrice en mode lecture seule
 * 
 * @tparam T Type générique à choisir
 * @param i Indice de ligne de matrice
 * @param j Indice de ligne de matrice
 * @return const T& Retourne la valeur à l'indice d'une matrice donnée
 */
template<typename T>
const T& nonOptMat<T>::operator()(int i, int j) const {
    return nonOptdata[i][j];
}

/**
 * @brief Surcharge de l'opérateur + pour l'addition entre deux matrices
 * 
 * @tparam T Type générique à choisir
 * @param A Matrice à additionner
 * @return Mat<T> Retourne une matrice étant l'addition des deux matrices
 */
template<typename T>
nonOptMat<T> nonOptMat<T>::operator+(const nonOptMat<T> &A) {
    
    if(size() != A.nonOptdata.size()){
        throw std::invalid_argument("Les matrices doivent être de même taille !");
    }

    nonOptMat<T> result(size());

    for(int i = 0; i < size(); i++){
        for(int j = 0 ; j < size(); j++) {
            result(i,j) = (*this)(i,j) + A(i,j);
        }
    }

    return result;
}

/**
 * @brief Surcharge de l'opérateur * pour le produit au sens matriciel
 * 
 * @tparam T Type générique à choisir
 * @param A Matrice à multiplier
 * @return Mat<T> Retourne une matrice étant le produit au sens matriciel des deux matrices
 */
template<typename T>
nonOptMat<T> nonOptMat<T>::operator*(const nonOptMat<T>& A) {

    nonOptMat<T> result(size(),0);

    for(int i = 0; i < size(); i++){
        for(int j = 0; j < size(); j++) {
            for(int k = 0; k < size(); k++) {
                result(i,j) += (*this)(i,k) * A(k,j); 
            }
        }
    }
    return result;
}

/**
 * @brief Fonction permettant de retourner la taille d'une matrice
 * 
 * @tparam T Type générique à choisir
 * @return int Retourne l'entier correspondant à la taille de la matrice
 */
template<typename T>
int nonOptMat<T>::size() {
    return nonOptdata.size();
}


/**
 * @brief Fonction permettant de remplir les valeurs d'une matrice avec des nombres aléatoires suivant une plage donnée
 * 
 * @tparam T Type générique à choisir
 * @param downRange Paramètre correspondant à la plage basse
 * @param UpRange Paramètre correspondant à la plage haute
 */
template<typename T>
void nonOptMat<T>::fillRandNumber(const T downRange, const T UpRange) {
    
    if constexpr (std::is_same<T,int>::value) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distrib(downRange, UpRange);

        for(int i = 0; i < size(); i++){
            for(int j = 0; j < size(); j++){
                (*this)(i,j) = distrib(gen);
            }
        }
    }

    else if constexpr (std::is_same<T,float>::value){

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(downRange, UpRange);

        for(int i = 0; i < size(); i++){
            for(int j = 0; j < size(); j++){
                (*this)(i,j) = distrib(gen);
            }
        }
    }

    else if constexpr (std::is_same<T,double>::value){

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distrib(downRange, UpRange);

        for(int i = 0; i < size(); i++){
            for(int j = 0; j < size(); j++){
                (*this)(i,j) = distrib(gen);
            }
        }
    }
}

/**
 * @brief Fonction permettant de réaliser un produit de Hadamard entre deux matrices
 * 
 * @tparam T Type générique à choisir
 * @param A Paramètre étant la seconde matrice à multiplier
 * @return Mat<T> Retourne le resultat du produit de Hadamard dans une matrice donnée
 */
template<typename T>
nonOptMat<T> nonOptMat<T>::hadamardProduct(const nonOptMat<T> &A) {

    if((*this).size() != A.nonOptdata.size()) {
        throw std::invalid_argument("Erreur Hadamard Product : Les matrices doivent avoir la même taille");
    }

    nonOptMat<T> result(size());

    for (int i = 0; i < size(); i++) {
        for (int j = 0; j < size();j++) {
            result(i,j) = (*this)(i,j) * A(i,j);
        }
    }
    
    return result;
}

