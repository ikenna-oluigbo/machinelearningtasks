import numpy as np 
import os
import csv
from sklearn.manifold import TSNE 


def file_path():
    filepath_emb = ['./embs/ASNE_eu_email_emb.csv']
    filepath_graph = './embs/twitter.txt' 
    model_name = 'DANE'
    return (filepath_emb, filepath_graph, model_name)

def transform_data():
    print('Unpacking Vector Embeddings...')
    for path in file_path()[0]:
        extension = os.path.splitext(path)[-1].lower()
        #Use this for .EMB files
        if extension == '.emb':
            np_emb = np.loadtxt(open(file_path()[0][-1]), skiprows = 1)
            nodes = [int(x[:1]) for x in np_emb]
            sliced_np_emb = np.array([x[1:] for x in np_emb])
            print('Rendering Embeddings into 2D...')
            tsne_model = TSNE(n_components = 2, perplexity = 30, learning_rate = 200.0)       #For larger datasets, Perplexity = 30 to 50
            Two_Dim_embs = tsne_model.fit_transform(sliced_np_emb)
            return (sliced_np_emb, Two_Dim_embs, nodes)
        
        #Use this for .CSV files
        elif extension == '.csv':    
            with open(file_path()[0][-1], 'r') as file:
                np_emb = list(csv.reader(file, delimiter=","))
            nodes = [x[:1] for x in np_emb]
            sliced_np_emb = np.array([x[1:] for x in np_emb[1:]])
            print('Rendering Embeddings into 2D...')
            tsne_model = TSNE(n_components = 2, perplexity = 30, learning_rate = 200.0)       #For larger datasets, Perplexity = 30 to 50
            Two_Dim_embs = tsne_model.fit_transform(sliced_np_emb)
            return (sliced_np_emb, Two_Dim_embs, nodes)  
            
            
        else:
            print("Error: Filetype needs to have .emb or .csv extension")