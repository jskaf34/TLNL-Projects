##### CODE #####

def ngram(name_f,n): #création d'un modèle ngram
    """
    Paramètres :
    name_f = chemin du fichier sur lequel calculer le modèle ngram
    n = grammage (n= 1, 2 ou 3)

    Output :
    le modèle ngram sous forme de dictionnaire
    """
    with open(name_f,"r", encoding='utf_8') as f_main :
        lignes = f_main.readlines()
        nb_ligne_test_prog = len(lignes)
        test_prog = lignes[:nb_ligne_test_prog]
        dict_ngram = {}
        for ligne in test_prog :
            mots_ligne = ligne.split()
            ngrams_ligne = []
            if n == 1 :
                ngrams_ligne = mots_ligne[:] #copie de la ligne
            else :
                for i in range(len(mots_ligne)-n+1):
                    ngrams_ligne.append(tuple(mots_ligne[i:i+n])) #tous les tuples de taille n pour chaque ligne
            for ngram in ngrams_ligne :
                if ngram in dict_ngram :
                    dict_ngram[ngram]+=1
                else :
                    dict_ngram[ngram]=1
    f_main.close()
    return dict_ngram

def ngram_list(name_f) :
    """
    Paramètres :
    name_f = chemin du fichier ou de la liste de fichiers sur lesquels calculer les modèles uni, bi et trigram

    Output :
    les modèles uni, bi et trigram sous forme d'une liste de dictionnaires
    """
    if type(name_f) == str : #un seul fichier
        list_ngram = [ngram(name_f,1), ngram(name_f,2), ngram(name_f,3)]

    elif type(name_f) == list : #un corpus, une liste de fichiers
        corpus = name_f
        list_ngram = [ngram(corpus[0],1), ngram(corpus[0],2), ngram(corpus[0],3)]
       
       #création du modèle final par concaténation des modèles générés sur chaque texte
        for i in range(1, len(corpus)):
            list_ngram_corpus_i = [ngram(corpus[i],1), ngram(corpus[i],2), ngram(corpus[i],3)]
            for j in range(len(list_ngram_corpus_i)) : 
                for key in list_ngram_corpus_i[j].keys() :
                    if key in list_ngram[j].keys():
                        list_ngram[j][key] += list_ngram_corpus_i[j][key] # ajout de la valeur à la clé déjà existante
                    else : 
                        list_ngram[j][key] = list_ngram_corpus_i[j][key] # ajout d'une nouvelle clé et de sa valeur

    return list_ngram


def write_ngram(name_f, name_new_f):
    """
    Paramètres :
    name_f = chemin du fichier ou de la liste de fichiers sur lesquels calculer les modèles uni, bi et trigram
    name_new_f = chemin du nouveau fichier contenant les modèles
    Output :
    la création de ce nouveau fichier
    """
    ngram = ngram_list(name_f)
    unigram = ngram[0]
    bigram = ngram[1]
    trigram = ngram[2]

    # Ouverture du fichier en mode écriture
    with open(name_new_f, 'w', encoding = 'utf_8') as fichier:

        fichier.write(f"#unigram {len(unigram)} \n\n") # nombre de valeurs dans l'unigram
        for cle, valeur in sorted(unigram.items(), key=lambda x: x[1], reverse=True): #dans l'ordre décroissant de fréquence d'apparition
            ligne = f"{cle} : {valeur}\n"
            fichier.write(ligne)

        fichier.write(f"#bigram {len(bigram)}\n\n")
        for cle, valeur in  sorted(bigram.items(), key=lambda x: x[1], reverse=True):
            ligne = f"{cle} : {valeur}\n"
            fichier.write(ligne)

        fichier.write(f"#trigram {len(trigram)}\n\n")
        for cle, valeur in sorted(trigram.items(), key=lambda x: x[1], reverse=True):
            ligne = f"{cle} : {valeur}\n"
            fichier.write(ligne)

    return f"Le dictionnaire a été écrit dans le fichier '{name_new_f}'."


##### TESTS #####


path= "C:/Users/emend/OneDrive/Documents/3A/TLNL/tlnl2/TP1_Mendizabal_Skaf/"
file = path + '/textes/alexandre_dumas/la_reine_margot.train.tok'
file2 = path + '/textes/alexandre_dumas/le_comte_de_monte_cristo.train.tok'
new_file = path + 'tests/ngram_la_reine_margot.txt'

# print('unigram sur "la reine Margot" :')
# print(ngram(file, 1))

# print('modèles ngram sur la concaténation de "la reine Margot" et "le comte de Monte Cristo:')
# print(ngram_list([file, file2]))

# print('génération du fichier contenant le modèle ngram de "la reine Margot":')
# print(write_ngram(file, new_file))

