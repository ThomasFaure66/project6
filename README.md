# project6

Ce répertoire regroupe l'ensemble du code utilisé pour le projet 6 du cous de simulation numérique d'ensemble statistique. Il porte sur a molécule diatomique HCl dans les approximation du potentiel Harmonique et potentiel de Morse. L'objectif est d'étudier la partition de l'énergie du système et le couplage rotatio vibration.

- Le fichier `main.py` contient le corps du code et fait appel aux autres fichiers pour obtenir des résultats physique.

- Le fichier `animations.py` contient un code permettant de générer et afficher une animation 3d de la molécule dans le cas sans thermostat.

- Le fichier `solver.py` contient les algorithmes de verlet avec et sans thermostat de Langevin. Ils fonctionne pour n'importe quelle force donc notament pour les 2 potentiels étudiés (Harmonique et Morse).

- Le fichier `potentials.py` contient les forces définies comme des fonctions des positions des atomes ainsi que les potentiels associés qui ous permettent de calculer l'énergie potentielle. S'il est executé, il génèrera égaement un graphe permettant de comparer les 2 potentiels.


