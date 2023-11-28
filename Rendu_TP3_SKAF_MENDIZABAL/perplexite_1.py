import torch
import numpy as np
import approfondissement_1
import learning



version = approfondissement_1.version
V = learning.V

parameters = torch.load(f'parameters_model_E{version}.pt')
W, U, b1, b2, E = parameters['W'], parameters['U'], parameters['b1'], parameters['b2'], parameters['E']
path_file_test = "Le_comte_de_Monte_Cristo.test.tok"
X_test, Y_test = approfondissement_1.initialisation(path_file_test,V)
index2word = learning.index2word
perplexite_temp = []




nb_mot_ok = 0
for i in range(len(X_test)):
    Y_pred = approfondissement_1.forward_inference(X_test[i], E, W, U, b1, b2)
    if Y_pred[Y_test[i]] > 10e-70 :
        nb_mot_ok +=1
        perplexite_temp.append(np.log(Y_pred[Y_test[i]].item()))
    else :
        perplexite_temp.append(0)

perplexite_temp = np.array(perplexite_temp)
perplexite = np.exp(np.sum(perplexite_temp)/(-len(Y_test)))
print(f'La perplexite est de {perplexite}')
print(f'nombre de mots différents de 0 : {nb_mot_ok/len(Y_test)*100}')



#comparer avec trigram markov
