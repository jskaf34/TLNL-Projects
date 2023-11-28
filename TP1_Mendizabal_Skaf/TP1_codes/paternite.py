##### CODE #####

from courbe_apprentissage import perplexite_on_models

def find_author(input_file, list_of_corpus, list_of_authors, n, LP=True):
    """
    Paramètres :
    input_file = nom du fichier dont on cherche l'auteur
    list_of_corpus = liste de plusieurs corpus, chacun associé à un auteur
    list_of_authors = liste des auteurs, dans le même ordre
    n = grammage
    LP = l'utilisation ou non de l'estimateur de Laplace (booleen)

    Output :
    Nom de l'auteur détecté, liste des perplexités associées à chaque corpus
    """
    list_perplexites = perplexite_on_models(list_of_corpus, input_file, n, LP)
    idx_author = list_perplexites.index(min(list_perplexites))
    
    return list_of_authors[idx_author], list_perplexites


##### TESTS #####

path = "C:/Users/emend/OneDrive/Documents/3A/TLNL/tlnl2/TP1_Mendizabal_Skaf/textes/"

D1 = path + 'alexandre_dumas/Le_comte_de_Monte_Cristo.tok'
D2 = path + 'alexandre_dumas/La_Reine_Margot.tok'
D3 = path + 'alexandre_dumas/Le_Vicomte_de_Bragelonne.tok'

B1 = path + 'honore_de_balzac/la_fille_aux_yeux_dor.tok'
B2 = path + 'honore_de_balzac/lelixir_de_longue_vie.tok'
B3 = path + 'honore_de_balzac/eugenie_grandet.tok'

V1 = path + 'jules_verne/autour_de_la_lune.tok'
V2 = path + 'jules_verne/aventures_du_capitaine_hatteras.tok'
V3 = path + 'jules_verne/face_au_drapeau.tok'

corpus_Dumas = [D2, D3]
corpus_Balzac = [ B2, B3]
corpus_Verne = [V2, V3]
list_models = [corpus_Dumas, corpus_Balzac, corpus_Verne]
list_authors = ['Dumas', 'Balzac', 'Verne']

print("Perplexite sur D1 : ", find_author(D1, list_models, list_authors, 3))
print("Perplexite sur B1 : ", find_author(B1, list_models, list_authors, 3))
print("Perplexite sur B1 : ", find_author(V1, list_models, list_authors, 3))

