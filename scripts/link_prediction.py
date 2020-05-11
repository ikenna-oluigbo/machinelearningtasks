import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
import numpy as np
import pandas as pd
from numpy.random import seed
import random 
import networkx as nx

from ml_main import transform_data, file_path

train_param = 0.9   #Tunable Training Parameters from 50% to 90%

def link_predict():
    (_, emb, nd) = transform_data()
    nodes = nd; embs = emb
    nodes.pop(0); print()
    print('Embedding Length: ', len(embs))
    print('Number of Nodes: ', len(nodes)); print(); print('Training Embeddings...')
    
    i = 1; meantest_list = []; meantrain_list = []
    while i < 11:
        print('Starting Iteration Number: ', i)
        train_limit = int(len(embs) * train_param)       
        data_train = embs[:train_limit]
        data_test = embs[train_limit:] 
                     
        nodes_limit = int(len(nodes) * train_param)
        node_train = nodes[:nodes_limit]
        node_test = nodes[nodes_limit:] 
                 
        x_train = np.array(pd.DataFrame(np.array([np.array(k) for k in data_train])))
        y_train = np.array(node_train)
    
        x_test = np.array(pd.DataFrame(np.array([np.array(k) for k in data_test])))   
        y_test = np.array(node_test)
                
        tf.set_random_seed(2020)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))
                     
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) 
        model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test), verbose = 1)    
        model.summary()
                     
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose = 1)
        train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose = 1)
        meantest_list.append(test_accuracy); meantrain_list.append(train_accuracy)
        i += 1
        
    with open('./embs/linkprediction.txt', mode = 'a') as lp:
            print('WALKLETS link Prediction at', str(int(train_param*100)) + '% Train Data', file = lp) 
            print('Training Accuracy for each run: ', meantrain_list, ' | ', 'Train Mean: ', np.mean(meantrain_list), file = lp) 
            print('Testing Accuracy for each run: ', meantest_list, ' | ', 'Test Mean: ', np.mean(meantest_list), file = lp) 
    
    yhat_probs = model.predict(embs, verbose=1)
    yhat_classes = model.predict_classes(embs, verbose=1)
    return (yhat_classes, embs)
link_predict()


def txt_2_dict(fn):
    with open(fn, 'r+') as k:
        l = k.readlines()
        l = [s.rstrip('\n') for s in l]
        first = []; second = []
        for path in l:
            f, s = path.split(sep = ' ')
            first.append(int(f)); second.append(int(s))
      
    edge_list = []; j_dict = {}
    [edge_list.append((first[i], second[i])) for i in range(len(first))]
      
    [j_dict[p[0]].add(p[1]) if p[0] in list(j_dict.keys()) 
        else j_dict.update({p[0]: {p[1]}}) for p in edge_list]
    return j_dict


deleted_edges = 1 - train_param
filename = file_path()[1]

def link_prediction_measure(): 
    print('Starting Heuristics Evaluation......')
    scores = [nx.resource_allocation_index, nx.jaccard_coefficient, nx.adamic_adar_index, nx.preferential_attachment]  
    dt = txt_2_dict(filename); pq = []
    ed = [{k:(k, v)} for k, val in dt.items() for v in val]
    pre_measure = int(len(ed) * deleted_edges)
    random.seed(2020)
    used_ed = random.sample(ed, pre_measure)
    used_ed_nodes = [kk for l in used_ed for kk, vv in l.items()]
    used_ed_edges = [vv for l in used_ed for kk, vv in l.items()]
    G = nx.Graph()
    G.add_nodes_from(used_ed_nodes)
    G.add_edges_from(used_ed_edges)
    for i in range(len(scores)):
        name = str(scores[i])
        preds = scores[i](G)
        #Generator Objects can only be accessed once. preds is a generator object in shape [(from_node, to_node, predictions from_node|to_node), ... (from,to, predctions)]
        for j in list(preds):
            pq.append(list(j)) 
        pred_queue = sorted([xx[2] for xx in pq if xx[2] != 0.0])
        print('Heuristic Scores for {}: '.format(name), np.mean([np.percentile(pred_queue, 100), np.percentile(pred_queue, 20)]))


#score = nx.preferential_attachment
#score = nx.resource_allocation_index
#score = nx.jaccard_coefficient
#score = nx.adamic_adar_index
#link_prediction_measure()