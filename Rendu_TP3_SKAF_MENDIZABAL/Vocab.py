import torch
import numpy as np

class Vocab:
    def __init__(self, fichier_matrice):
        self.dico_voca = {}
        with open(fichier_matrice,'r') as fi:
            ligne = fi.readline()
            ligne = ligne.strip()
            
            #self.emb_dim, self.vocab_size = eval(ligne)
            self.vocab_size, self.emb_dim = map(int,ligne.split(" "))
            self.matrice = torch.zeros((self.vocab_size, self.emb_dim))
            indice_mot = 0
        
            ligne = fi.readline()
            ligne = ligne.strip()
            while ligne != '': 
            
                splitted_ligne = ligne.split()
                self.dico_voca[splitted_ligne[0]] = indice_mot
                for i in range(1,len(splitted_ligne)):
                    self.matrice[indice_mot, i-1] = float(splitted_ligne[i])
                indice_mot += 1
                ligne = fi.readline()
                ligne = ligne.strip()

    def get_word_index(self, mot):
        if not mot in self.dico_voca:
            return 2
        return self.dico_voca[mot]
                
    def get_emb(self, mot):
        if not mot in self.dico_voca:
            return None
        return  self.matrice[self.dico_voca[mot]]
    
    def get_emb_torch(self, indice_mot):
        # OPTIMISATION: no verificaiton allows to get embeddings a bit faster
        #if indice_mot < 0 or indice_mot >= self.matrice.shape()[0]: # not valid index
        #    return None
        #return self.matrice[indice_mot]
        return self.matrice[indice_mot]
        
    def get_one_hot(self, mot):
        vect = torch.zeros(len(self.dico_voca))
        vect[self.dico_voca[mot]] = 1
        return vect

