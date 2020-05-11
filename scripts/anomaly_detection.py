import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager
import pandas as pd

from pyod.models.iforest import IForest
from pyod.models.knn import KNN

#from ml_main import transform_data
from link_prediction import link_predict

#embs = transform_data()[1] 
(yhat_classes, embeds) = link_predict()
embs = embeds

df = pd.DataFrame(embs[0:])
df.columns = ['first_column', 'second_column']

col_1 = df['first_column'].values.reshape(-1,1)
col_2 = df['second_column'].values.reshape(-1,1)
concat_df = np.concatenate((col_1,col_2),axis=1) 

ce_list = []

print('Starting Anomaly Detection Algorithms......')
random_state = np.random.RandomState(2020)
outliers_fraction = 0.01
Anomaly_Detection_Algorithms = { 
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),  
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state), 
}

xx, yy = np.meshgrid(np.linspace(concat_df.min(), concat_df.max(), 500), np.linspace(concat_df.min(), concat_df.max(), 500))           

for i, (algorithm_name, algo) in enumerate(Anomaly_Detection_Algorithms.items()):
    algo.fit(concat_df)
    scores_pred = algo.decision_function(concat_df) * -1
    datapoint_pred = algo.predict(concat_df)
    num_of_inliers = len(datapoint_pred) - np.count_nonzero(datapoint_pred)
    num_of_outliers = np.count_nonzero(datapoint_pred == 1) 
    
    plt.figure(figsize=(11, 11))
    
    new_df = df
    new_df['outlier_column'] = datapoint_pred.tolist()
    
    Inlier_Feature_Space1 = np.array(df['first_column'][new_df['outlier_column'] == 0]).reshape(-1,1)
    Inlier_Feature_Space2 = np.array(df['second_column'][new_df['outlier_column'] == 0]).reshape(-1,1)
    
    Outlier_Feature_Space1 = df['first_column'][new_df['outlier_column'] == 1].values.reshape(-1,1)
    Outlier_Feature_Space2 = df['second_column'][new_df['outlier_column'] == 1].values.reshape(-1,1)
         
    print('ALGORITHM:',algorithm_name, ' | ', 'NUMBER OF INLIERS: ',num_of_inliers, ' | ', 'NUMBER OF OUTLIERS: ',num_of_outliers)
    
    inlier_outlier_threshold = np.percentile(scores_pred,100 * outliers_fraction)
            
    raw_score = algo.decision_function(np.c_[xx.reshape(-1), yy.reshape(-1)]) * -1
    raw_score = raw_score.reshape(xx.shape)
    raw_score_min = raw_score.min()
    
    if inlier_outlier_threshold < raw_score_min:
        inlier_outlier_threshold, raw_score_min = raw_score_min, inlier_outlier_threshold

    plt.contourf(xx, yy, raw_score, levels=np.linspace(raw_score_min, inlier_outlier_threshold, 10),cmap=plt.cm.Blues_r)
    plt.contourf(xx, yy, raw_score, levels=[inlier_outlier_threshold, raw_score.max()],colors='orange')
    
    a = plt.contour(xx, yy, raw_score, levels=[inlier_outlier_threshold],linewidths=1, colors='red')     
    b = plt.scatter(Inlier_Feature_Space1, Inlier_Feature_Space2, c='white', s=20, edgecolor='black')
    c = plt.scatter(Outlier_Feature_Space1, Outlier_Feature_Space2, c='black', s=20, edgecolor='black')
    
    plt.autoscale(enable=True, axis='x', tight=True)
    
    plt.legend(title = 'SPLITTER-Num_Inliers | Num_Outliers = ' + str(num_of_inliers) + ' | ' + str(num_of_outliers),
        handles=[a.collections[0], b, c],
        labels=['learned decision function', 'inliers', 'outliers'],  
        prop=matplotlib.font_manager.FontProperties(size=10),
        loc='upper right')
      
    plt.xlim((concat_df.min(), concat_df.max()))
    plt.ylim((concat_df.min(), concat_df.max()))
    plt.title(algorithm_name)
    print('Saving file to disk......') 
    plt.show()
    
    '''Evaluating Anomaly Detection Results'''
    yc = yhat_classes
    epsilon=1e-12
    
    predictions = np.clip(scores_pred, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(datapoint_pred*np.log(predictions+1e-9))/N 
    
    with open('./embs/CE.txt', mode = 'a') as loss:
        print('Anomaly Detection Cross-Entropy loss SPLITTER Twitter:', round(ce,4), file = loss)
