# BackwardForwardSokoban
Implementation of the paper "Solving Sokoban with Forward-backward Reinforcement Learning" by Y. Shoham &amp; G. Elidan

Install `gym-sokoban` from source.
The `raw` representation of the states is only available on the latest versions.


TODO:
* Penser à distinguer le gamma des features et celui pour l'entraînement
* Checker à chaque étape si les deux gammas sont bien employés, actuellement c'est le bordel
* Changer les rewards pour avoir uniquement -1 ou 1 en fonction de la victoire ou de la défaite (défaite = deadlock) ?
* Il faut que l'état final ait une value de 0
