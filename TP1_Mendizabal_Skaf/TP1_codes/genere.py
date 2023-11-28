
##### CODE #####

from ngram import ngram_list


def generate_sentence_unigram(sorted_word_dict, sentence, d):
    """
    Paramètres :
    sorted_word_dict = modèle unigram, sous forme de dictionnaire trié par ordre décroissant de fréquence d'apparition
    sentence = début de phrase générée à compléter (inutile ici, mais utile pour fonctions homologues récursives bi et trigram)
    d = nombre de mots désirés dans la phrase générée

    Output :
    liste des d mots de la phrase générée avec modèle unigram
    """
    sentence = list(sorted_word_dict.keys())[:d]
    return sentence


def generate_sentence_bigram(sorted_word_dict, sentence, d): #fonction récursive
    """
    Paramètres :
    sorted_word_dict = modèle bigram, sous forme de dictionnaire trié par ordre décroissant de fréquence d'apparition
    sentence = début de la phrase générée à compléter
    d = nombre de mots désirés dans la phrase générée

    Output :
    liste des d mots de la phrase générée avec modèle unigram
    """

    if d == 1: #condition intiale
        return sentence
    
    new_word = sentence[-1] #dernier mot ajouté à la phrase, utile pour trouver le prochain si elle est encore incomplète
    found = False
    for (m1, m2), v in sorted_word_dict.items() : 
        if m1 == new_word and m2 not in sentence: #on évite de rajouter le même mot en boucle
            sentence.append(m2)
            del sorted_word_dict[(m1, m2)]
            found = True
            break
    
    if found : 
        return generate_sentence_bigram(sorted_word_dict, sentence, d-1) #nouveau mot ajouté, on continue en diminuant le nb de mots nécessaires
    
    else : #pas de correspondance entre le dernier mot de la phrase incomplète et les bigrams. On ajoute donc un mot au hasard avec bonne probabilité.
        for (m1, m2), v in sorted_word_dict.items() : 
            if m2 not in sentence: 
                sentence.append(m2) 
                del sorted_word_dict[(m1, m2)]
                return generate_sentence_bigram(sorted_word_dict, sentence, d-1)
            
    
def generate_sentence_trigram(sorted_word_dict, sentence, d):
    """
    Même principe que generate_sentence_bigram
    """

    if d == 2: 
        return sentence
    
    new_word_1, new_word_2 = sentence[-2], sentence[-1]
    found = False
    for (m1, m2, m3), v in sorted_word_dict.items() : 
        if m1 == new_word_1 and m2 == new_word_2 and m3 not in sentence: 
            sentence.append(m3)
            del sorted_word_dict[(m1, m2, m3)]
            found = True
            break
    
    if found : 
        return generate_sentence_trigram(sorted_word_dict, sentence, d-1)
    else : 
        for (m1, m2, m3), v in sorted_word_dict.items() : 
            if m3 not in sentence: 
                sentence.append(m3)
                del sorted_word_dict[(m1, m2, m3)]
                return generate_sentence_trigram(sorted_word_dict, sentence, d-1)            


def generate_sentence(name_f, n, d): 
    """
    Paramètres :
    path_f =  nom du fichier ou de la liste de fichiers sur lequels créer le modèle
    n = grammage
    d = nombre de mots désirés dans la phrase générée

    Output :
    la phrase de d mots, générée avec modèle ngram
    """

    #Preparation du modèle
    ngram = ngram_list(name_f)
    unigram = ngram[0]
    bigram = ngram[1]
    trigram = ngram[2]

    if n == 1: 
        prefix = []
        sorted_ngram_dict = dict(sorted(unigram.items(), key=lambda item: item[1], reverse=True)) #création du dictionnaire trié
        sentence = generate_sentence_unigram(sorted_ngram_dict, prefix, d) #liste des mots générés
    elif n ==2: 
        prefix = ['<s>']
        sorted_ngram_dict = dict(sorted(bigram.items(), key=lambda item: item[1], reverse=True))
        sentence = generate_sentence_bigram(sorted_ngram_dict, prefix, d) #ici le préfixe est nécessaire, de même que pour trigram
    elif n ==3: 
        prefix = ['<s>', '<s>']
        sorted_ngram_dict = dict(sorted(trigram.items(), key=lambda item: item[1], reverse=True))
        sentence = generate_sentence_trigram(sorted_ngram_dict, prefix, d)

    sentence_str = ' '.join(sentence)
    return sentence_str


##### TESTS #####

path= "C:/Users/emend/OneDrive/Documents/3A/TLNL/tlnl2/TP1_Mendizabal_Skaf/textes/"
file = path + 'alexandre_dumas/la_reine_margot.train.tok'
n = 2
d = 20

# print(f'génération de phrases de {d} mots à partir du texte "La Reine Margot"')
# print("unigram : ", generate_sentence(file, 1, d))
# print("bigram : ", generate_sentence(file, 2, d))
# print("trigram : ", generate_sentence(file, 3, d))