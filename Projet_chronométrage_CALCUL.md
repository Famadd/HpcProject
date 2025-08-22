**Projet chronométrage du temps de calcul du produit matriciel/addition matriciel/produit de Hadamard**

Le projet a pour but de produire des algorithmes visant à faire des opérations de calcul sur des matrices plus ou moins grandes, ainsi qu'à calculer leur temps d'exécution pour pouvoir les comparer.


Il se fera sous CPU dans un premier temps et sur GPU dans un second temps.

Ce projet est une sorte d'initiation à l'informatique HPC ainsi qu'à la programmation Multi-thread via l'API CUDA.


Vous pourrez voir le comparatif des performances des différents programmes au bas de ce README.

Le projet tourne sous C++20 et plus.
Pour pouvoir compiler et exécuter ces programmes (Surtout pour la partie GPU), il vous faudra installer les dépendances suivantes :

    L'API CUDA Toolkit 13.0 et plus, incluant le compilateur nécessaire nvcc ainsi que cuBLAS.



Installation du CUDA Toolkit 13.0
    https://developer.nvidia.com/cuda-downloads


Passons au descriptif des algorithmes.

**CPU**


ALGORITHME 1 :

 	Calculs matriciels (Addition/Produit matriciel/Produit de Hadamard) avec un algorithme naïf (càd non optimisé)

ALGORITHME 2 :

    Calculs matriciels (Addition/Produit matriciel/Produit de Hadamard) avec un algorithme version linéaire, optimisé sur l'accession des données sur le CPU (accès contiguë des tableaux).



**GPU**



ALGORITHME 3 :

    Calculs matriciels (Addition/Produit matriciel/Produit de Hadamard) avec un algorithme version linéaire, optimisé sur l'accession des données sur le GPU.


**Partie CPU I3 12100f 32GO de RAM**



ALGORITHME 1 :


Matrice N\*N 			Avec N = 1000

Temps de calcul (Produit matriciel) = 17.0654s

Temps de calcul (Addition de matrice) = 0.0170739s

Temps de calcul (Produit de Hadamard) = 0.0157567s





Matrice N\*N 			Avec N = 10 000

Temps de calcul (Produit matriciel) = 23660,4s ~= 6h34m

Temps de calcul (Addition de matrice) = 1.56241s

Temps de calcul (Produit de Hadamard) = 1.57841s





ALGORITHME 2 :



Matrice N\*N			Avec N = 1000

Temps de calcul (Produit matriciel) = 0.0113341s

Temps de calcul (Addition de matrice) = 0.0112113s

Temps de calcul (Produit de Hadamard) = 0.0111302s



Matrice N\*N			Avec N = 10000

Temps de calcul (Produit matriciel) = 1.19419s

Temps de calcul (Addition de matrice) = 1.14798s

Temps de calcul (Produit de Hadamard) = 1.15504s



**Partie GPU GTX 1660 Super TURING TU116 - 1408 Cœurs CUDA 6 Go VRAM**

ALGORITHME 3 :


Matrice N\*N			Avec N = 1000

Temps de calcul (produit matriciel) = 0.00489494 s

Temps de calcul (Addition de matrice) = 0.00181629 s

Temps de calcul (Produit de Hadamard) = 0.000231424 s



Matrice N\*N			Avec N = 10000

Temps de calcul (produit matriciel) = 0.64972 s

Temps de calcul (Addition de matrice) = 0.0117779 s

Temps de calcul (Produit de Hadamard) = 0.00953139 s







