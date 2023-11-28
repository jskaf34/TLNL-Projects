##### CODE #####

from ngram import ngram_list
import numpy as np 


def nombre_mot(name_f):
    """
    Paramètres :
    name_f = nom du fichier sur lequel calculer le nombre de mots

    Output :
    le nombre de mots (int)
    """
    nb_mot = 0
    with open(name_f,'r', encoding='utf_8') as f :
        lignes = f.readlines()
        for ligne in lignes:
            mots = ligne.split()
            for i in range(len(mots)):
                nb_mot+=1
    return nb_mot

def nombre_mots(corpus): 
    """
    Paramètres :
    corpus = liste des fichiers sur lequels calculer le nombre de mots total

    Output :
    le nombre de mots total (int)
    """
    total_word_count = 0
    for file_path in corpus:
        total_word_count += nombre_mot(file_path)
    return total_word_count

def cardinal_V(unigram):
    """
    Paramètres :
    unigram = dictionnaire du modèle unigram

    Output :
    le nombre de mots composant le vocabulaire (int)
    """
    card_V = 0
    for _ in unigram.keys():
        card_V+=1
    return card_V

def proba_unigram(count_mot, card_V, N, LP):
    """
    Paramètres :
    count_mot = le nombre d'apparitions du mot dans le modèle unigram (int)
    card_V = le nombre de mots composants le vocabulaire dans le modèle unigram (int)
    N = le nombre de mots dans le texte étudié (int)
    LP = l'utilisation ou non de l'estimateur de Laplace (booleen)

    Output :
    la probabilité unigram (float)
    """
    if LP :
        return (count_mot+1)/(N + card_V)
    else :
        return (count_mot)/N

def proba_bigram(count_mot_m_m1,count_mot_m1, card_V, LP):
    """
    Paramètres :
    count_mot_m_m1 = le nombre d'apparitions du tuple (m,m1) dans le modèle bigram (int)
    count_mot_m1 = le nombre d'apparitions du mot m1 dans le modèle unigram (int)

    Output :
    la probabilité bigram (float)
    """
    if LP :
        return (count_mot_m_m1+1)/(count_mot_m1 + card_V)
    else :
        return (count_mot_m_m1)/(count_mot_m1)

def proba_trigram(count_mot_m_m1_m2,count_mot_m1_m2, card_V, LP):
    """
    Paramètres :
    count_mot_m_m1_m2 = le nombre d'apparitions du tuple (m,m1,M2) dans le modèle trigram (int)

    Output :
    la probabilité trigram (float)
    """
    if LP : 
        return (count_mot_m_m1_m2+1)/(count_mot_m1_m2 + card_V)
    else :
        return (count_mot_m_m1_m2)/(count_mot_m1_m2)



def perplexite(name_f_model, name_f_perp, n, LP=False):
    """
    Paramètres :
    name_f_model = nom du fichier ou de la liste de fichiers sur lequels créer le modèle
    name_f_perp = nom du fichier sur lequel calculer la perplexité
    n = grammage
    LP = l'utilisation ou non de l'estimateur de Laplace (booleen)

    Output :
    la perplexité (float)
    """
    #Calcul du modèle
    ngram = ngram_list(name_f_model)
    unigram = ngram[0]
    bigram = ngram[1]
    trigram = ngram[2]
    N = [nombre_mot(name_f_model) if type(name_f_model)==str else nombre_mots(name_f_model)][0]
    card_V = cardinal_V(unigram)
    N_perp = nombre_mot(name_f_perp)
    S = 0 #initialisation de la somme présente dans la log perplexité

    #Calcul de perplexité par ligne
    with open(name_f_perp,'r', encoding = 'utf_8') as fichier :
        mot_connu = 0
        lignes = fichier.readlines()
        for ligne in lignes:
            mots = ligne.split()
            for i in range(len(mots)):

                #unigram
                if n == 1:
                    try :
                        #Si le ngram est dans notre vocabulaire
                        count_mot = unigram[mots[i]]
                        mot_connu +=1
                    except KeyError:
                        # sinon on lui attribue 0
                        count_mot = 0
                    finally :
                        # calcul de la somme cummulée des log proba
                        S += np.log(proba_unigram(count_mot, card_V, N, LP))

                #bigram
                elif n == 2:
                    if i == 0:
                        pass
                    else :
                        try:
                            #Si le ngram est dans notre vocabulaire
                            count_mot_mot_moins_1 = bigram[(mots[i-1],mots[i])]
                            mot_connu +=1
                        except KeyError:
                            # sinon on lui attribue 0
                            count_mot_mot_moins_1 = 0
                        try :
                            count_mot_moins_1 = unigram[mots[i-1]]
                        except KeyError:
                            count_mot_moins_1 = 0 
                        finally :
                            # calcul de la somme cummulée des log proba
                            S += np.log(proba_bigram(count_mot_mot_moins_1,count_mot_moins_1, card_V, LP))

                #trigram
                elif n == 3:
                    if i == 0 or i == 1:
                        pass
                    else : 
                        try :
                            count_mot_mot_moins_1_mot_moins_2 = trigram[(mots[i-2],mots[i-1],mots[i])]
                            mot_connu +=1
                        except KeyError:
                            count_mot_mot_moins_1_mot_moins_2 = 0
                        try :
                            count_mot_moins_1_mot_moins_2 = bigram[(mots[i-2],mots[i-1])]
                        except KeyError:
                            count_mot_moins_1_mot_moins_2 = 0
                        finally :
                            proba = proba_trigram(count_mot_mot_moins_1_mot_moins_2,count_mot_moins_1_mot_moins_2, card_V, LP)
                            S += np.log(proba)
    return np.exp(S/(-N_perp))



##### TESTS #####

path= "C:/Users/emend/OneDrive/Documents/3A/TLNL/tlnl2/TP1_Mendizabal_Skaf/textes/"
file = path + 'alexandre_dumas/la_reine_margot.train.tok'
file2 = path + 'alexandre_dumas/le_comte_de_monte_cristo.train.tok'
n = 2
LP = True

# print('perplexité du comte de Monte Cristo pour un modèle entraîné sur la Reine Margot :')
# print(perplexite(file, file2, n, LP))



