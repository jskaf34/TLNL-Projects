import Vocab 
import torch
import numpy as np
import time

nb_V = 2908
k = 5 #nb_voc_prefixe + mot cible (donc dans cours, correspond à k+2)
d = 100
dim_h = 200
b = 8
epoch = 60
eta = 0.00005


path_file_vocab = "embeddings-word2vecofficial.train.unk5.txt"
path_file_train = "Le_comte_de_Monte_Cristo.train.unk5.tok"
V = Vocab.Vocab(path_file_vocab)


batch_on = 1
version = 1


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

def func_index2word(name_file_embedding,V):
    index2word = {}
    f_main = open(name_file_embedding,"r")
    lignes = f_main.readlines()
    for ligne in lignes[1:] :
        mots_ligne = ligne.split()
        if V.get_word_index(mots_ligne[0]) == 2:
            index2word[V.get_word_index(mots_ligne[0])] = '<unk>'
        else :
            index2word[V.get_word_index(mots_ligne[0])] = mots_ligne[0]
    return index2word

def create_input(liste_k_voisins):
    global V
    input = torch.tensor([])
    for j in range(len(liste_k_voisins)-1):
        input = torch.cat((input,V.get_emb_torch(liste_k_voisins[j])))
    return input

def create_output(Y):
    global V, index2word
    return V.get_one_hot(index2word[Y])

def create_input_batch(batch_liste_k_voisins):
    global V,b
    temp = []
    for i in range(b):
        temp.append(create_input(batch_liste_k_voisins[i]))
    return torch.stack(tuple(temp))

def create_output_batch(Y_batch):
    global V,b
    temp = []
    for i in range(b):
        temp.append(create_output(Y_batch[i]))
    return torch.stack(tuple(temp))

def forward_apprentissage(W, U, b1, b2,x):
    h = torch.nn.functional.relu(x @ W + b1)
    y_pred = h @ U + b2
    return y_pred 

def forward_inference(W, U, b1, b2,x):
    h = torch.nn.functional.relu(x @ W + b1)
    y_pred = torch.nn.functional.softmax(h @ U + b2)
    return y_pred 

def apprentissage_one_by_one(X,Y):
    global V, epoch, loss, optimizer, W, U, b1, b2, k, d
    for e in range(epoch):
        total_error = 0
        indices = np.random.randint(0,len(X),len(X))
        for i in indices :
            tx = create_input(X[i]) 
            ty = create_output(Y[i]) 
            Y_pred = forward_apprentissage(W, U, b1, b2,tx)
            error = loss(Y_pred,ty)
            error.backward()
            total_error += error.item()
            optimizer.step()
            optimizer.zero_grad()
        print(total_error/len(indices))


def apprentissage_batchs(X,Y):
    global V, epoch, loss, optimizer, W, U, b1, b2
    for e in range(epoch):
        print(f'----------------------------epoch {e}----------------------------')
        total_error = 0
        indices = np.random.randint(0,len(X),len(X))
        for i in indices :
            tx = create_input_batch(X[i]) 
            ty = create_output_batch(Y[i])# coder avec un long tensor pour ne pas faire toutes les multiplications
            Y_pred = forward_apprentissage(W, U, b1, b2,tx)
            error = loss(Y_pred,ty)
            error.backward()
            total_error += error.item()
            optimizer.step()
            optimizer.zero_grad()
        print(total_error/len(indices))

index2word = func_index2word(path_file_vocab,V)

if __name__ == "__main__":
    if batch_on:
        t_0 = time.time()
        X_batchs, Y_batchs = initialisation_batch(path_file_train,V)
        W = torch.rand(((k-2)*d,dim_h), requires_grad=True)
        b1 = torch.rand((dim_h), requires_grad=True)
        U = torch.rand((dim_h,nb_V), requires_grad=True)
        b2 = torch.rand((nb_V), requires_grad=True)
        optimizer = torch.optim.Adam((W, U, b1, b2), lr=eta)
        loss = torch.nn.CrossEntropyLoss()
        apprentissage_batchs(X_batchs,Y_batchs)
        parameters = {
        'W':W, 
        'U':U, 
        'b1':b1, 
        'b2':b2
    }
        torch.save(parameters,f"parameters_model_batch_V{version}.pt")
        elapsed_time = time.time() - t_0
        print(f'temps écoulé : {elapsed_time} s')
    else : 
        X, Y = initialisation(path_file_train,V)
        W = torch.rand(((k-2)*d,dim_h), requires_grad=True)
        b1 = torch.rand((dim_h), requires_grad=True)
        U = torch.rand((dim_h,nb_V), requires_grad=True)
        b2 = torch.rand((nb_V), requires_grad=True)
        optimizer = torch.optim.Adam((W, U, b1, b2), lr=eta)
        loss = torch.nn.CrossEntropyLoss()
        apprentissage_one_by_one(X,Y)
        parameters = {
        'W':W, 
        'U':U, 
        'b1':b1, 
        'b2':b2
    }
        torch.save(parameters,f"parameters_model_V{version}.pt")

