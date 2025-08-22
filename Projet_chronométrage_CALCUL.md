# 🚀 Projet : Chronométrage des calculs matriciels (CPU & GPU)

## 🎯 Objectif du projet
Le but de ce projet est d’implémenter différents algorithmes de calcul de matrices carrées afin de **comparer leurs performances** sur CPU et GPU.  
Les opérations testées sont :  
- **Addition de matrices**  
- **Produit matriciel**  
- **Produit de Hadamard**  

👉 Les temps d’exécution sont mesurés pour des matrices de taille croissante, afin d’évaluer l’impact de l’optimisation et du matériel utilisé.  

Ce projet est une **initiation à l’informatique HPC** et à la **programmation parallèle** via l’API CUDA.  

---

## 🖥️ Environnement et prérequis
- **Langage :** C++20 et plus  
- **Dépendances :**
  - [CMake](https://cmake.org/download/)
  - [CUDA Toolkit 13.0+](https://developer.nvidia.com/cuda-downloads)  
  - `nvcc` (compilateur CUDA)  
  - **cuBLAS** (inclus dans le CUDA Toolkit)  

## ⚡ Exécution du programme
  - Placez vous dans le build d'un des deux dossiers TestsCPU/TestsGPU
  - Entrez la commande *cmake..*
  - Vous trouverez l'exécutable dans le dossier bin/

---

## ⚙️ Description des algorithmes

### **CPU**
- **Algorithme 1 :** Implémentation naïve (non optimisée).  
- **Algorithme 2 :** Implémentation linéaire avec **accès mémoire contigu** (optimisé CPU).  

### **GPU**
- **Algorithme 3 :** Implémentation linéaire avec **optimisation GPU CUDA**.  

---

## 📊 Résultats expérimentaux

### CPU (Intel i3-12100F, 32 Go RAM)

#### Algorithme 1 (naïf)
| Taille N×N | Produit matriciel | Addition | Hadamard |
|------------|------------------:|---------:|----------:|
| 1000       | 17.0654 s         | 0.01707 s | 0.01575 s |
| 10000      | ~23660.4 s (~6h34m) | 1.56241 s | 1.57841 s |

#### Algorithme 2 (linéaire optimisé)
| Taille N×N | Produit matriciel | Addition | Hadamard |
|------------|------------------:|---------:|----------:|
| 1000       | 0.01133 s         | 0.01121 s | 0.01113 s |
| 10000      | 1.19419 s         | 1.14798 s | 1.15504 s |

---

### GPU (NVIDIA GTX 1660 Super — TU116, 1408 cœurs CUDA, 6 Go VRAM)

#### Algorithme 3 (CUDA optimisé, cuBLAS, algorithme de tiling pour le produit de Hadamard)
| Taille N×N | Produit matriciel | Addition | Hadamard |
|------------|------------------:|---------:|----------:|
| 1000       | 0.00489 s         | 0.00182 s | 0.00023 s |
| 10000      | 0.64972 s         | 0.01178 s | 0.00953 s |

---

## ✅ Conclusion
- L’algorithme naïf est **impraticable** pour de grandes tailles (ex. produit 10000×10000 prend **6h34m**).  
- L’optimisation CPU réduit drastiquement le temps (de **6h34m → ~1.2s**) grâce à l'accession linéaires des données.  
- Le GPU est encore plus performant pour les gros calculs, atteignant **0.65s** pour une matrice 10000×10000, avec la librairie optimisé cuBLAS.  

⏳ **GPU >> CPU optimisé >> CPU naïf**  
---

## 🔮 Améliorations possibles
- Implémenter le **tiling** et le **caching** pour les kernels CUDA, utile pour les très grosses matrices (Impossible à stocker en mémoire).  
- Implémenter une interface visuelle ergonomique comme **QT**.
- Étendre aux matrices non carrées.

## 👉 N'hésitez pas à me faire des retours si vous pensez que cela puisse m'aider/être intéressant ! 
