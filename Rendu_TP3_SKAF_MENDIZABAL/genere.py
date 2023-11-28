import torch
import numpy as np
import learning
import time
import tkinter as tk

k = learning.k
version = learning.version
V = learning.V
nb_V = learning.nb_V

index2word = learning.index2word
parameters = torch.load(f'parameters_model_batch_V{version}.pt')
W, U, b1, b2 = parameters['W'], parameters['U'], parameters['b1'], parameters['b2']
path_file_test = "Le_comte_de_Monte_Cristo.test.tok"
l = nb_V #pour la génération on échantillone sur les k plus grande valeur
teta = 1.5 #tempéature pour accentuer variabilité ou pic de probabilité dans la distribution prédite

def generer_mot(Y_pred):
    global l, teta

    choisi = 0
    Y_pred_generation, index = torch.topk(Y_pred, l)
    Y_pred_generation = Y_pred_generation/teta
    while not(choisi):
        indice_echantillonage = np.random.randint(len(Y_pred_generation))
        p = np.random.random()
        if p < Y_pred_generation[indice_echantillonage]:
            index_mot_genere = index[indice_echantillonage].item()
            choisi = 1
    return index_mot_genere

        
def afficher_suite(proba_mot_attendu, index_proba_mot_attendu, proba_mot_plus_probable, index_proba_mot_plus_probable,proba_mot_genere,index_proba_mot_genere):
    global index2word, index
    mon_text.insert("end", f" Mot attentdu : {index2word[index_proba_mot_attendu[index]]}\n Probabilité du mot attendu : {proba_mot_attendu[index]}\n Mot le plus probable : {index2word[index_proba_mot_plus_probable[index]]}\n Probabilité du mot le plus probable : {proba_mot_plus_probable[index]}\n Mot prédit : {index2word[index_proba_mot_genere[index]]}\n Probabilité du mot prédit : {proba_mot_genere[index]}\n\n")
    index +=1


X_test, Y_test = learning.initialisation(path_file_test,V)
proba_mot_attendu = []
index_proba_mot_attendu = []
proba_mot_plus_probable = []
index_proba_mot_plus_probable = []
proba_mot_genere= []
index_proba_mot_genere = []


for i in range(len(X_test)):
    tx = learning.create_input(X_test[i]) 
    ty = learning.create_output((Y_test[i]))
    Y_pred = learning.forward_inference(W, U, b1, b2,tx)
    proba_mot_attendu.append(Y_pred[Y_test[i]].item())
    index_proba_mot_attendu.append(Y_test[i])
    proba_mot_plus_probable.append(torch.max(Y_pred).item())
    index_proba_mot_plus_probable.append(torch.argmax(Y_pred).item())
    index_mot_genere = generer_mot(Y_pred)
    proba_mot_genere.append(Y_pred[index_mot_genere].item())
    index_proba_mot_genere.append(index_mot_genere)

debut = [index2word[X_test[0][j]] for j in range(k-1)]
index=0
fenetre = tk.Tk()
fenetre.title("Affichage de print dans une fenêtre")
mon_text = tk.Text(fenetre)
mon_text.pack()
texte_statique = str(debut)
mon_text.insert("end", texte_statique + "\n")
bouton_afficher_dans_fenetre = tk.Button(fenetre, text="Afficher les prints dans la fenêtre", command= lambda : afficher_suite(proba_mot_attendu, index_proba_mot_attendu, proba_mot_plus_probable, index_proba_mot_plus_probable,proba_mot_genere,index_proba_mot_genere))
bouton_afficher_dans_fenetre.pack()
fenetre.mainloop()



