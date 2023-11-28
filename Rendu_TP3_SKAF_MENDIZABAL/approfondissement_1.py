import Vocab 
import torch
import numpy as np
import learning
# import time



nb_V = 2908
d = 100
k = 5 #nb_voc_prefixe + mot cible (donc dans cours, correspond à k+2)
dim_h = 100
b = 8
epoch = 60
eta = 0.0001
version = 8

path_file_vocab = "embeddings-word2vecofficial.train.unk5.txt"
path_file_train = "Le_comte_de_Monte_Cristo.train.unk5.tok"
V = Vocab.Vocab(path_file_vocab)
index2word = learning.index2word

#j'utilise seulement indiçage de V pour ne pas refaire un unigram etc.
def make_E_SGNS():
    global V, nb_V, d, index2word
    E = torch.zeros(nb_V, d, requires_grad=True)
    with torch.no_grad():  
        for key, value in index2word.items():
            E[key] = V.get_emb_torch(key)
    return E

def initialisation(name_file,V):
    global k
    f_main = open(name_file,"r")
    lignes = f_main.readlines()
    X = []
    Y = []
    for ligne in lignes :
        mots_ligne = ligne.split()
        for i in range(len(mots_ligne)-k):
            X.append([V.get_word_index(mots_ligne[i + j]) for j in range(k-1)])
            Y.append(V.get_word_index(mots_ligne[i + k - 1]))
    f_main.close()
    return X,Y


def initialisation_batch(name_file,V):
    global k, b
    f_main = open(name_file,"r")
    lignes = f_main.readlines()
    X = []
    Y = []
    for ligne in lignes :
        mots_ligne = ligne.split()
        for i in range(len(mots_ligne)-k):
            X.append([V.get_word_index(mots_ligne[i + j]) for j in range(k-1)])
            Y.append(V.get_word_index(mots_ligne[i + k - 1]))
    indice_batch_random = np.random.randint(0,len(X),len(X))
    batch_X = []
    batch_Y = []
    X_batchs = []
    Y_batchs = []
    for i in indice_batch_random:
        if len(batch_Y) < b :
            batch_X.append(X[i])
            batch_Y.append(Y[i])
        else :
            X_batchs.append(batch_X)
            Y_batchs.append(batch_Y)
            batch_X = []
            batch_Y = []
            batch_X.append(X[i])
            batch_Y.append(Y[i])
    if batch_Y  !=  Y_batchs[-1]:
        Y_batchs.append(batch_Y)
    f_main.close()
    return X_batchs,Y_batchs


def create_input(liste_k_voisins, E):
    global V
    input = torch.tensor([])
    for j in range(len(liste_k_voisins)-1):
        input = torch.cat((input,E[liste_k_voisins[j]]))
    return input

def create_input_batch(batch_liste_k_voisins, E):
    global V,b
    temp = []
    for i in range(b):
        temp.append(create_input(batch_liste_k_voisins[i],E))
    return torch.stack(tuple(temp))

def create_output(Y):
    global V, index2word
    return V.get_one_hot(index2word[Y])


def forward_apprentissage(X, E, W, U, b1, b2):
    x = create_input_batch(X,E)
    h = torch.nn.functional.relu(x @ W + b1)
    y_pred = h @ U + b2
    return y_pred 


# def forward_apprentissage_residu(X, E, W, U, b1, b2, beta):
#     x = create_input_batch(X,E)
#     h = torch.nn.functional.relu(x @ W + b1)
#     y_pred = h @ U + b2 + E @ beta # beata fait la moyenne pondérée sur les dimension de E
#     return y_pred 

def forward_inference(X_i, E, W, U, b1, b2):
    x = create_input(X_i,E)
    h = torch.nn.functional.relu(x @ W + b1)
    y_pred = torch.nn.functional.softmax(h @ U + b2)
    return y_pred 

# def forward_inference_residu(X_i, E, W, U, b1, b2, beta):
#     x = create_input(X_i,E)
#     h = torch.nn.functional.relu(x @ W + b1)
#     y_pred = torch.nn.functional.softmax(h @ U + b2 + E @ beta)
#     return y_pred 

def create_output_batch(Y_batch):
    global V,b
    temp = []
    for i in range(b):
        temp.append(create_output(Y_batch[i]))
    return torch.stack(tuple(temp))


def apprentissage_batchs(X,Y):
    global V, epoch, loss, optimizer, W, U, b1, b2, E
    for e in range(epoch):
        print(f'----------------------------epoch {e}----------------------------')
        total_error = 0
        indices = np.random.randint(0,len(X),len(X))
        for i in indices :
            ty = create_output_batch(Y[i])  # coder avec un long tensor pour ne pas faire toutes les multiplications
            Y_pred = forward_apprentissage(X[i], E, W, U, b1, b2)
            error = loss(Y_pred,ty)
            error.backward()
            total_error += error.item()
            optimizer.step()
            optimizer.zero_grad()
        print(total_error/len(indices))




if __name__ == "__main__":
    # t_0 = time.time()
    X_batchs, Y_batchs = initialisation_batch(path_file_train,V)
    E = torch.rand((nb_V,d), requires_grad=True)
    #E = make_E_SGNS()
    W = torch.rand(((k-2)*d,dim_h), requires_grad=True)
    b1 = torch.rand((dim_h), requires_grad=True)
    U = torch.rand((dim_h,nb_V), requires_grad=True)
    b2 = torch.rand((nb_V), requires_grad=True)
    optimizer = torch.optim.Adam((W, U, b1, b2, E), lr=eta)
    loss = torch.nn.CrossEntropyLoss()
    apprentissage_batchs(X_batchs,Y_batchs)
    parameters = {
    'W':W, 
    'U':U, 
    'b1':b1, 
    'b2':b2,
    'E': E
}
    torch.save(parameters,f"parameters_model_E{version}.pt")
#     elapsed_time = time.time() - t_0
#     print(f'temps écoulé : {elapsed_time} s')