##### CODE #####

from ngram import ngram_list
import numpy as np 
from perplexite import cardinal_V, proba_unigram, proba_bigram, proba_trigram, nombre_mot, nombre_mots



def add_unk(unigram, bigram, trigram, seuil_unk):
    """
    Paramètres :
    unigram, bigram, trigram = modèles ngram générés sous forme de dictionnaires
    seuil_unk = valeur limite pour le compteur de mots dits "unknown". 

    Output :
    aucun output mais le compte de "unknown" est ajouté à chaque dictionnaire. 
    si dans unigram, le mot est <unk> alors dans bigram et trigram il est également considéré <unk>
    """

    #initialisation
    unigram['<unk>'] = 0
    bigram['<unk>'] = 0
    trigram['<unk>'] = 0

    for key, value in unigram.items():
        if value <= seuil_unk :
            unigram['<unk>'] += 1
       
    for key, value in bigram.items():
        if value <= seuil_unk :
            bigram['<unk>'] += 1

    for key, value in trigram.items():
        if value <= seuil_unk :
            trigram['<unk>'] += 1
    

def devine_unigram(unigram, seuil_unk, N, card_V, LP=True): 
    """
    Output :
    liste des mots avec la meilleure probabilité d'apparition dans le texte à trous
    """

    #Initialisation
    max_proba = 0
    words = []

    for key, value in unigram.items():
        if key == '<s>' or key == '</s>':
            continue
        #valeur de la proba : 
        if value <= seuil_unk:
            proba_key = proba_unigram(unigram['<unk>'], N, card_V, LP)
        else :
            proba_key = proba_unigram(value, N, card_V, LP)
        #comparaison des probas et ajout ou non du mot à la liste: 
        if proba_key > max_proba:
            max_proba = proba_key
            words = []
            words.append(key)
        elif proba_key == max_proba:
            words.append(key)

    return words


def devine_bigram(sentence, position_word, bigram, unigram, seuil_unk, card_V, LP=True): 
    """
    Paramètres :
    sentence = phrase étudiée
    position_word = index du mot à deviner

    Output :
    liste des mots avec la meilleure probabilité d'apparition dans le texte à trous
    """
    possibilities = {}
    words = []
    for key in unigram.keys():
        if key == '<s>' or key == '</s>':
            continue
        try:
            #Si le ngram est dans notre vocabulaire
            count_mot_mot_moins_1 = bigram[(sentence[position_word-1],key)]
        except KeyError:
            # sinon on lui attribue 0
            count_mot_mot_moins_1 = 0
        try :
            count_mot_moins_1 = unigram[sentence[position_word-1]]
        except KeyError:
            count_mot_moins_1 = 0 
        try:
            #Si le ngram est dans notre vocabulaire
            count_mot_mot_plus_1 = bigram[(key,sentence[position_word+1])]
        except KeyError:
            # sinon on lui attribue 0
            count_mot_mot_plus_1 = 0
        try :
            count_mot = unigram[key]
        except KeyError:
            count_mot = 0 
        finally :
           possibilities[key] = proba_bigram(count_mot_mot_moins_1,count_mot_moins_1, card_V, LP) * proba_bigram(count_mot_mot_plus_1,count_mot, card_V, LP)
    
    max_proba = 0
    for key, value in possibilities.items(): 
        if value > max_proba:
            max_proba = value
            words = []
            if unigram[key] <= seuil_unk:
                key = '<unk>'
            words.append(key)
        elif value == max_proba:
            if unigram[key] <= seuil_unk:
                key = '<unk>'
            words.append(key)
    return words


def devine_trigram(sentence, position_word, trigram, bigram, unigram, seuil_unk, card_V, LP=True): 
    """
    Même chose que bigram, version trigram
    """
    possibilities = {}
    words = []
    for key in unigram.keys():
        if key == '<s>' or key == '</s>':
            continue
        try :
            count_mot_mot_moins_1_mot_moins_2 = trigram[(sentence[position_word-2],sentence[position_word-1],key)]
        except KeyError:
            count_mot_mot_moins_1_mot_moins_2 = 0
        try :
            count_mot_moins_1_mot_moins_2 = bigram[(sentence[position_word-2],sentence[position_word-1])]
        except KeyError:
            count_mot_moins_1_mot_moins_2 = 0
        try :
            count_mot_moins_1_mot_mot_plus_1 = trigram[(sentence[position_word-1],key,sentence[position_word+1])]
        except KeyError:
            count_mot_moins_1_mot_mot_plus_1 = 0
        try :
            count_mot_moins_1_mot = bigram[(sentence[position_word-1],key)]
        except KeyError:
            count_mot_moins_1_mot = 0
        try:
            count_mot_mot_plus_1_mot_plus_2 = trigram[(key, sentence[position_word+1],sentence[position_word+2])]
        except KeyError:
            count_mot_mot_plus_1_mot_plus_2 = 0
        try:
            count_mot_plus_1_mot_plus_2 = bigram[(key,sentence[position_word+1])]
        except KeyError:
            count_mot_plus_1_mot_plus_2 = 0
        possibilities[key] = proba_trigram(count_mot_mot_moins_1_mot_moins_2,count_mot_moins_1_mot_moins_2, card_V, LP) * proba_trigram(count_mot_moins_1_mot_mot_plus_1,count_mot_moins_1_mot, card_V, LP) * proba_trigram(count_mot_mot_plus_1_mot_plus_2,count_mot_plus_1_mot_plus_2, card_V, LP)
    
    max_proba = 0
    for key, value in possibilities.items(): 
        if value > max_proba:
            max_proba = value
            words = []
            if unigram[key] <= seuil_unk:
                key = '<unk>'
            words.append(key)
        elif value == max_proba:
            if unigram[key] <= seuil_unk:
                key = '<unk>'
            words.append(key)
    return words

def score_devine(name_f_model, name_f_devine, n, seuil_unk):
    """
    Paramètres :
    name_f_model = nom du fichier ou de la liste de fichiers sur lequels créer le modèle
    name_f_perp = nom du fichier sur lequel calculer les performances de remplissage des trous
    n = grammage
    valeur limite pour le compteur de mots dits "unknown". 

    Output :
    score de remplissage des trous, entre 0 (aucun mot trouvé) et 1 (tous les mots trouvés)
    """ 

    #Calcul du modèle
    ngram = ngram_list(name_f_model)
    unigram = ngram[0]
    bigram = ngram[1]
    trigram = ngram[2]
    add_unk(unigram, bigram, trigram, seuil_unk)
    N = [nombre_mot(name_f_model) if type(name_f_model)==str else nombre_mots(name_f_model)][0]
    card_V = cardinal_V(unigram)

    #Calcul des performances sommées sur chaque ligne
    if n == 1:
        well_guessed = 0
        with open(name_f_devine,'r', encoding = 'utf_8') as fichier :
            lignes = fichier.readlines()
            nb_lignes = len(lignes)
            for ligne in lignes:
                mots = ligne.split()
                position = np.random.randint(0, len(mots)) #pour les textes non masqués
                # position = int(mots[0]) +1 #pour les textes masqués 
                words = devine_unigram(unigram, seuil_unk, N, card_V)
                if len(words)==1:
                    guess = words[0]
                else :
                    index = np.random.randint(0, len(words))
                    guess = words[index]
                target = mots[position]
                try :
                    unigram[target]
                except KeyError:
                    target = '<unk>'
                if unigram[target] <= seuil_unk:
                    target = '<unk>'
                if target == guess: #mot trouvé
                    well_guessed+=1
            well_guessed/=100
           
            return well_guessed

    elif n == 2:
        well_guessed = 0
        with open(name_f_devine,'r', encoding = 'utf_8') as fichier :
            lignes = fichier.readlines()
            nb_lignes = len(lignes)
            for ligne in lignes:
                mots = ligne.split()
                position = np.random.randint(0, len(mots)-1)  #pour les textes non masqués
                # position = int(mots[0]) + 1  #pour les textes masqués
                words = devine_bigram(mots,position,bigram,unigram, seuil_unk, card_V)
                if len(words)==1:
                    guess = words[0]
                else :
                    index = np.random.randint(0, len(words))
                    guess = words[index]
                target = mots[position]
                try :
                    unigram[target]
                except KeyError:
                    target = '<unk>'
                if unigram[target] <= seuil_unk:
                    target = '<unk>'
                if target == guess:
                    well_guessed+=1
            well_guessed/=100
            return well_guessed

    elif n == 3:
        well_guessed = 0
        with open(name_f_devine,'r', encoding='utf_8') as fichier :
            lignes = fichier.readlines()
            nb_lignes = len(lignes)
            for ligne in lignes:
                mots = ligne.split()
                position = np.random.randint(0, len(mots)-2)  #pour les textes non masqués
                # position = int(mots[0]) + 1  #pour les textes masqués
                words = devine_trigram(mots,position,trigram,bigram,unigram, seuil_unk, card_V)
                if len(words)==1:
                    guess = words[0]
                else :
                    index = np.random.randint(0, len(words))
                    guess = words[index]
                target = mots[position]
                try :
                    unigram[target]
                except KeyError:
                    target = '<unk>'
                if unigram[target] <= seuil_unk:
                    target = '<unk>'
                if target == guess:
                    well_guessed+=1
            well_guessed/=100
            return well_guessed

    else :
        return 'Erreur dans la saisie de n'


##### TESTS #####

path = "C:/Users/emend/OneDrive/Documents/3A/TLNL/tlnl2/TP1_Mendizabal_Skaf/textes/"

corpus1 = [path + 'alexandre_dumas/Le_comte_de_Monte_Cristo.tok', path + 'alexandre_dumas/La_Reine_Margot.tok', path + "alexandre_dumas/Les_Trois_Mousquetaires.tok", 
           path + 'alexandre_dumas/Le_Vicomte_de_Bragelonne.tok', path + 'alexandre_dumas/Vingt_ans_apres.tok']

corpus2 = [path + 'honore_de_balzac/la_fille_aux_yeux_dor.tok', path + 'honore_de_balzac/lelixir_de_longue_vie.tok', 
           path + 'honore_de_balzac/eugenie_grandet.tok']

corpus3 = [path + 'jules_verne/autour_de_la_lune.tok', path + 'jules_verne/aventures_du_capitaine_hatteras.tok',path + 'jules_verne/cinq_semaines_en_ballon.tok',
           path + 'jules_verne/face_au_drapeau.tok',path + 'jules_verne/larchipel_en_feu.tok', path + 'jules_verne/le_docteur_ox.tok'
           , path + 'jules_verne/le_tour_du_monde_en_80_jours.tok', path + "jules_verne/le_pilote_du_danube.tok"]

corpus4 = [path + 'wikipedia/ia_wikipedia.tok', path + 'wikipedia/science_wikipedia.tok']

corpus_full = corpus1 + corpus2 + corpus3 + corpus4
input_file = path + 'marie_lebert/le_projet_gutemberg.tok'
n = 3
seuil_unk = 5

# print('la proportion de mot bien deviné pour le modèle unigram est de ', score_devine(corpus_full, input_file, n, seuil_unk))


