##### CODE #####

from perplexite import perplexite
from devine import score_devine
from genere import generate_sentence
import numpy as np
import plotly.graph_objs as go


def perplexite_on_models(list_corpus, input_file, n, LP=True):
    """
    Paramètres :
    list_corpus = la liste des sous-corpus, par taille croissante 
    (1er élément = 1er texte, 2ème élément = concaténation des 1er et 2ème textes, etc)
    input_file = fichier dont on veut calculer la perplexité
    n = grammage
    LP = l'utilisation ou non de l'estimateur de Laplace (booleen)

    Output :
    la liste des perplexités par sous-corpus (float)
    """
    list_perplexites = []
    for corpus in list_corpus :
        list_perplexites.append(perplexite(corpus, input_file, n, LP))
    return list_perplexites

def devine_on_models(list_corpus, input_file, n, seuil_unk): 
    """
    Même chose pour les performances de remplissage de texte à trou
    """
    list_devine = []
    for corpus in list_corpus :
        list_devine.append(score_devine(corpus, input_file, n, seuil_unk))
    return list_devine

def genere_on_models(list_corpus, n, d): 
    """
    Même chose pour la génération de phrase de d mots
    """
    list_genere = []
    for corpus in list_corpus :
        list_genere.append(generate_sentence(corpus, n, d))
    return list_genere


def list_corpus_diff_size(corpus): 
    """
    Paramètres :
    corpus = liste de textes

    Output :
    la liste des sous-corpus, par taille croissante 
    (1er élément = 1er texte, 2ème élément = concaténation des 1er et 2ème textes, etc)
    """

    #Initialisation
    list_corpus = []
    concat_corpus = []
    
    for file in corpus:
        concat_corpus.append(file)
        list_corpus.append(concat_corpus.copy())  # copie pour ne pas modifier les entrées précédentes

    return list_corpus


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

categories = np.arange(len(corpus_full)) + np.ones(len(corpus_full))
categories = [int(c) for c in categories]
list_corpus = list_corpus_diff_size(corpus_full)
values = perplexite_on_models(list_corpus, input_file, 2, 20)

# # Create a bar chart using Plotly Express
# fig = go.Figure(data=go.Scatter(x=categories, y=values, mode='lines+markers', name='Values'))

# # Update the layout for better presentation (optional)
# fig.update_layout(
#     title="Performances texte à trou sur 'le projet Gutemberg' selon le nb de textes dans la concaténation de tous les corpus",
#     xaxis_title='Nb de textes',
#     yaxis_title='Peformance (between 0 and 1)')

# # Show the plot
# fig.show()




