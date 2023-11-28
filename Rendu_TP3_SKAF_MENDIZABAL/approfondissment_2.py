import torch
import numpy as np
import learning
import approfondissement_1
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

nb_V = 2908
d = 100
K = 20
index2word = learning.func_index2word(learning.path_file_vocab,learning.V)

def similarite(m1,m2):
    sim = torch.dot(m1,m2)/(torch.linalg.vector_norm(m1)*torch.linalg.vector_norm(m2))
    return sim.item()

def make_E_SGNS():
    global V, nb_V, d, index2word
    E = torch.zeros(nb_V,d)
    for key, value in index2word.items():
        E[key] = V.get_emb_torch(key)
    return E


#Paramètres utiles pour le fichier
version = approfondissement_1.version
V = approfondissement_1.V
parameters = torch.load(f'parameters_model_E{version}.pt')
W, U, b1, b2, E = parameters['W'], parameters['U'], parameters['b1'], parameters['b2'], parameters['E']
path_file_sim = "Le_comte_de_Monte_Cristo.test.tok"
eval_similarites = []
neighborhood_of_words = []
common_neighborhood_of_words = []
nb_common_neighborhood_of_words = []
E_SGNS = make_E_SGNS()




#Tâche de similarité
with open("Le_comte_de_Monte_Cristo.100.sim",'r') as evaluation :
    lignes = evaluation.readlines()
    for ligne in lignes:
        ligne = ligne.split()
        eval_E = int(similarite(E[V.get_word_index(ligne[0])], E[V.get_word_index(ligne[1])]) > similarite(E[V.get_word_index(ligne[0])], E[V.get_word_index(ligne[2])]))
        eval_SGNS = int(similarite(V.get_emb(ligne[0]), V.get_emb(ligne[1])) > similarite(V.get_emb(ligne[0]), V.get_emb(ligne[2])))
        eval_similarites.append([eval_E,eval_SGNS])



eval_similarites_np = np.array(eval_similarites)
data_eval_similarites = (pd.DataFrame({
    "Method" : ["E_learned" for i in range(100)] + ["E_SGNS" for i in range(100)],
    "Value_sim": np.concatenate((eval_similarites_np[:,0],eval_similarites_np[:,1]))
}
    )
    .groupby("Method").mean()
    .rename(columns={"Value_sim":"Value_sim_mean"}))


sns.barplot(data=data_eval_similarites, x=data_eval_similarites.index, y="Value_sim_mean")
plt.show()



#Nombre de voisinage commun
for i in range(nb_V):
    index_mot_cible = i
    neighborhood_cible_SGNS = set(torch.topk(-torch.linalg.norm(E_SGNS-E_SGNS[i], dim=1), k=K)[1].tolist())
    neighborhood_cible_lr = set(torch.topk(-torch.linalg.norm(E-E[i], dim=1), k=K)[1].tolist())
    neighborhood_of_words.append([neighborhood_cible_SGNS,neighborhood_cible_lr])
    common_neighborhood_of_words.append(neighborhood_cible_SGNS.intersection(neighborhood_cible_lr))
    nb_common_neighborhood_of_words.append(len(neighborhood_cible_SGNS.intersection(neighborhood_cible_lr)))
nb_common_neighborhood_of_words = np.array(nb_common_neighborhood_of_words) - 1

data_common_neighborhood_of_words = pd.DataFrame({
    "Nb Common Neighborhood" : nb_common_neighborhood_of_words 
}           
                ,index=index2word.values())

sns.set(style="darkgrid")
sns.barplot(data = data_common_neighborhood_of_words, x= data_common_neighborhood_of_words.index, y= "Nb Common Neighborhood", lw=0.)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.show()



#Projection t_SNE
perplexities_tsne = [0.001, 0.01, 0.1,5,10,50,100,150,200,500]
fig, axs = plt.subplots(2, len(perplexities_tsne))

for i  in range(len(perplexities_tsne)):
    perplexity = perplexities_tsne[i]
    E_embedded_i = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(E.detach().numpy())
    data_E_embedded_i = pd.DataFrame({
    "x" : E_embedded_i[:,0],
    "y": E_embedded_i[:,1]
})
    sns.scatterplot(data = data_E_embedded_i, x="x", y="y", ax=axs[0,i])
    print(f"Graphe E TSE {i+1} done !")

for i  in range(len(perplexities_tsne)):
    perplexity = perplexities_tsne[i]
    E_SGNS_embedded_i = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(E_SGNS.detach().numpy())
    data_E_SGNS_embedded_i = pd.DataFrame({
    "x" : E_SGNS_embedded_i[:,0],
    "y": E_SGNS_embedded_i[:,1]
})
    sns.scatterplot(data = data_E_SGNS_embedded_i, x="x", y="y", ax=axs[1,i])
    print(f"Graphe E_SGNS TSE {i+1} done !")
    

plt.show()


#cacher les axes









