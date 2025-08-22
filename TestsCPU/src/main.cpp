#include "optimisedCalculus.cpp"
#include "NonOptimisedCalculus.cpp"


int main() {

    int choix;
    int taille;
    do{
        std::cout << "Programme comparatif entre algorithmes calculatoires d'operations sur matrices" << std::endl;
        std::cout << "Choissisez l'algorithme de calcul matriciel que vous voulez\n" << std::endl;
        std::cout << "1 - Algorithme Non optimise\n2 - Algorithme Optimise \n";
        std::cin >> choix;
    } while( choix != 1 && choix != 2 );
    do{
        std::cout << "choissisez maintenant la taille des matrices que vous voulez calculez" << std::endl;
        std::cout << "Celle-ci doit etre un entier entre 0 et 100 000" << std::endl;
        std::cin >> taille;

    } while ( taille > 0 && taille > 100000 );
    

    if(choix == 1){
        std::cout << "Calcul Matriciel non optimise ---------------------------" << std::endl;

        nonOptimisedMatricialProductTime(taille, 0, 25);
        nonOptimisedMatricialAdditionTime(taille, 0, 2500.250f);
        nonOptimisedMatricialHadamardProductTime(taille, 0 ,10000.150);

        std::cout << "\nFin Calcul Matriciel non optimise ---------------------------" << std::endl;
    }

    if(choix == 2){
        std::cout << "Calcul Matriciel optimise ---------------------------" << std::endl;
        
        optimisedMatricialProductTime(taille, 0, 25);

        optimisedMatricialAdditionTime(taille, 0, 2500.250f);

        optimisedMatricialHadamardProductTime(taille, 0 ,15000.150);

        std::cout << "Fin Calcul Matriciel optimise ---------------------------" << std::endl;
    }

    return 0;
}
