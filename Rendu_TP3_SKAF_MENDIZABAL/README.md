Ce dossier contient notre implémentation du TP3 sur les modèles de langage neuronaux.

Il s'organise ainsi:

* un fichier learning.py qui contient les fonctions qui permettent d'apprendre un MLN, avec et sans minibatch, pour E figé et intialisé avec la méthode SGNS.
Il suffit de setup à 1 la variable "batch" pour passe en apprentissage par mini-batch. Les hyperparamètres sont similaires à ceux introduits dans le rapport. Pour charger un modèle préexistant, il suffit de glisser un fichier .pt dans le dossier principal. S'il s'agit de paramètres pour l'entrainement du vrai modèle (i.e qui n'a pas sevi aux experimentations des hyperparamètres), il suffit d'initialiser la variable "version" au bon nombre correspondant. Exemple : pour un fichier "parameters_model_batch_V0.pt", il suffit de le mettre dans le dossier principal et mettre version = 0 pour le charger.

* un fichier perplexite.py qui permet de calculer la perplexité d'un modèle sur le fichier test

* un fichier genere.py qui permet de générer du texte à partir des premières phrases du fichier test. Il s'affiche dans une fenêtre et il suffit de cliquer sur le bouton pour afficher la suite. Les hyperparamètres l et alpha permettent de considérer les l mots les plus probables à générer et alpha est une température pour moduler la variété de la génération.

* des fichiers approfondissement_1.py, perplexite_1.py et genere_1.py qui permettent d'apprendre un MLN en incluant E dans l'apprentissage. Le fonctionnement est identique respectivement aux fichiers au-dessus. Les paramètres du MLN sont stockés dans des fichiers de type "parameters_model_batch_E{k}.pt"

* un fichier approfondissement_2.py qui implémente les tâches et graphiques décrits la piste 2 du rapport.

* un dossier avec les figures utilisées pour le rapport

* un dossier contenant tous les paramètres des modèles utilisés ou testés pour notre rapport.

A titre d'indication :  

* les paramètres de modèle utilisé pour la perplexité et la génération à E figé et initialisé avec SGNS sont sauvegardés dans le fichier parameters_model_batch_V1.pt.

* les paramètres de modèle utilisé pour la perplexité et la génération à E figé et initialisé aléatoirement sont sauvegardés dans le fichier parameters_model_E4.pt.

* les paramètres de modèle utilisé pour la perplexité et la génération à E appris et initialisé aléatoirement sont sauvegardés dans le fichier parameters_model_E6.pt.

* les paramètres de modèle utilisé pour la perplexité et la génération à E appris et initialisé avec SGNS sont sauvegardés dans le fichier parameters_model_E8.pt.


