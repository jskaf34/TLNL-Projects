##### CODE ####

# Fonction pour masquer les mots dans une phrase
def mask_words(sentence, list_masks):
    return ['' if word in list_masks else word for word in sentence]

def masque_noms_propres(name_f, name_f_mask, list_masks) : 

    # Lire le fichier d'entrée et stocker les phrases
    with open(name_f, "r", encoding = 'utf_8') as f:
        sentences = [line.split() for line in f]

    # Masquer les mots choisis dans les phrases
    masked_sentences = [mask_words(sentence, list_masks) for sentence in sentences]


    # Écrire les phrases masquées dans le fichier de sortie
    with open(name_f_mask, "w", encoding='utf_8') as f_mask:
        for sentence in masked_sentences:
            f_mask.write(' '.join(sentence) + '\n')
    f.close()
    f_mask.close()


##### TESTS #####

path = "C:/Users/emend/OneDrive/Documents/3A/TLNL/tlnl2/TP1_Mendizabal_Skaf/textes/"
inputFileName = path + 'alexandre_dumas/le_comte_de_Monte_Cristo.train.tok'
outputFileName = path + 'tests/masque_monte_cristo.tok'
wordsToMask = ['monte', 'cristo', 'dantès', 'albert', 'villefort']
# masque_noms_propres(inputFileName, outputFileName, wordsToMask)
# print("Fichier généré avec les mots masqués")

