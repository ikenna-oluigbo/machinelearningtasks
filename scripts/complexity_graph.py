import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

#Model Values from link prediction accuracies
def AUCROC_linkpredict():
    df=pd.DataFrame({'x': range(50,100,10), 'dane': (0.6664,0.6806,0.7011,0.7344,0.7788), 'bane': (0.6233,0.6918,0.7157,0.7885,0.8056), 
    'deepwalk': (0.6677,0.6710,0.7824,0.8036,0.8478), 'fscnmf': (0.6707,0.7081,0.7208,0.7943,0.8200), 'graphwave': (0.7047,0.7286,0.7801,0.8030,0.8435), 
    'grarep': (0.6744,0.6842,0.7089,0.7108,0.7221), 'node2vec': (0.6226,0.7108,0.7942,0.8195,0.8509), 'tadw': (0.6236,0.6438,0.6711,0.7833,0.7910), 
    'hivec': (0.7311,0.7712,0.7825,0.8063,0.8402), 'walklets': (0.6362,0.6761,0.6959,0.7288,0.7590), 'boostedne': (0.6015,0.6784,0.7079,0.7450,0.7482), 
    'danmf': (0.7173,0.7328,0.7909,0.8007,0.8168), 'diff2vec': (0.6165,0.7051,0.7409,0.7864,0.8317), 'gemsec': (0.6262,0.6691,0.6944,0.7342,0.7577), 
    'musae': (0.6246,0.7025,0.7271,0.7869,0.8196), 'prune': (0.6162,0.6326,0.6410,0.7147,0.7193), 'role2vec': (0.6826,0.7022,0.7152,0.7627,0.7803), 
    'sine': (0.6177,0.6616,0.7465,0.7799,0.8446), 'splitter': (0.6239,0.6506,0.7043,0.7517,0.7644), 'line': (0.5942,0.6511,0.6996,0.7458,0.7701)})
    plt.plot('x', 'dane', data=df, marker='*', color='red', linewidth=1, label="dane")  #linestyle='dashed'
    plt.plot('x', 'bane', data=df, marker='*', color='green', linewidth=1, label="bane")
    plt.plot('x', 'deepwalk', data=df, marker='*', color='blue', linewidth=1, label="deepwalk")
    plt.plot('x', 'fscnmf', data=df, marker='*', color='yellow', linewidth=1, label="fscnmf")
    plt.plot('x', 'graphwave', data=df, marker='*', color='black', linewidth=1, label="graphwave")
    plt.plot('x', 'grarep', data=df, marker='*', color='purple', linewidth=1, label="grarep")
    plt.plot('x', 'node2vec', data=df, marker='*', color='chocolate', linewidth=1, label="node2vec")
    plt.plot('x', 'tadw', data=df, marker='*', color='grey', linewidth=1, label="tadw")
    plt.plot('x', 'hivec', data=df, marker='*', color='orange', linewidth=1, label="hivec")
    plt.plot('x', 'walklets', data=df, marker='*', color='pink', linewidth=1, label="walklets")
    plt.plot('x', 'boostedne', data=df, marker='*', color='brown', linewidth=1, label="boostedne")
    plt.plot('x', 'danmf', data=df, marker='*', color='darkblue', linewidth=1, label="danmf")
    plt.plot('x', 'diff2vec', data=df, marker='*', color='violet', linewidth=1, label="diff2vec")
    plt.plot('x', 'gemsec', data=df, marker='*', color='olive', linewidth=1, label="gemsec")
    plt.plot('x', 'musae', data=df, marker='*', color='gold', linewidth=1, label="musae")
    plt.plot('x', 'prune', data=df, marker='*', color='darkorange', linewidth=1, label="prune")
    plt.plot('x', 'role2vec', data=df, marker='*', color='aqua', linewidth=1, label="role2vec")
    plt.plot('x', 'sine', data=df, marker='*', color='springgreen', linewidth=1, label="sine")
    plt.plot('x', 'splitter', data=df, marker='*', color='coral', linewidth=1, label="splitter")
    plt.plot('x', 'line', data=df, marker='*', color='maroon', linewidth=1, label="line")
    
    plt.ylabel('ROC')
    plt.xlabel('Train Ratios')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=8, mode="expand", borderaxespad=0.)
    
    plt.show()
#AUCROC_linkpredict()


#Dissimilarity Index Values calculated from Cluster Centroids
def clustering_plot():
    df=pd.DataFrame({'model': ('DANE','BANE','DEEPWALK','FSCNMF','GRAPHWAVE','GRAREP','NODE2VEC','TADW','HIVEC','WALKLETS','BOOSTEDNE',
                               'DANMF','DIFF2VEC','GEMSEC','MUSAE','PRUNE','ROLE2VEC','SINE','SPLITTER'), 
                    'dissimilarity_index': (1.94, 2.80, 2.76, 1.26, 2.85, 4.16, 2.18, 2.17, 2.23, 2.45, 2.95,
                                            1.90, 2.45, 1.72, 3.94, 6.55, 3.36, 3.68, 3.86)})
    list_diss = list(df['dissimilarity_index'])
    plt.figure(figsize=(10, 10), dpi=65)
    for i in range(len(list_diss)):
        #plt.plot('model', 'dissimilarity_index', data=df, marker='s', color='blue', linewidth=1, linestyle='dashed', label="Dissimilarity Proximity Index")
        plt.bar('model', 'dissimilarity_index', data=df)
        plt.annotate(list_diss[i], xy = (i, list_diss[i]), xytext = (0, 0), textcoords='offset points', ha='right', va='bottom')
    
    plt.ylabel('Dissimilarity Index')
    plt.show()
#clustering_plot()


#Model Values from number of outliers
def anomaly_bar():
    algo = ['iForest', 'KNN']
    DANE = [26, 9]; BANE = [15, 17]; DEEPWALK = [27, 24]; FSCNMF = [28, 13]; GRAPHWAVE = [9, 16]; GRAREP = [27, 17]; NODE2VEC = [27, 19]
    TADW = [28, 16]; HIVEC = [28, 18]; WALKLETS = [24, 21]; BOOSTEDNE = [16, 26]; DANMF = [27, 15]; DIFF2VEC = [24, 17]; GEMSEC = [26, 19]
    MUSAE = [25, 21]; PRUNE = [27, 11]; ROLE2VEC = [25, 12]; SINE = [28, 16]
    
    ind = np.arange(2)
    width = 0.04
    
    plt.bar(ind, DANE, width, color='red', label='DANE')
    plt.bar(ind+width, BANE, width, color='green', label='BANE')
    plt.bar(ind+(2*width), DEEPWALK, width, color='blue', label='DEEPWALK')
    plt.bar(ind+(3*width), FSCNMF, width, color='brown', label='FSCNMF')
    plt.bar(ind+(4*width), GRAPHWAVE, width, color='black', label='GRAPHWAVE') 
    plt.bar(ind+(5*width), GRAREP, width, color='purple', label='GRAREP') 
    plt.bar(ind+(6*width), NODE2VEC, width, color='magenta', label='NODE2VEC') 
    plt.bar(ind+(7*width), TADW, width, color='springgreen', label='TADW') 
    plt.bar(ind+(8*width), HIVEC, width, color='coral', label='HIVEC') 
    plt.bar(ind+(9*width), WALKLETS, width, color='pink', label='WALKLETS')
    plt.bar(ind+(10*width), BOOSTEDNE, width, color='olive', label='BOOSTEDNE') 
    plt.bar(ind+(11*width), DANMF, width, color='violet', label='DANMF')  
    plt.bar(ind+(12*width), DIFF2VEC, width, color='grey', label='DIFF2VEC')
    plt.bar(ind+(13*width), GEMSEC, width, color='orange', label='GEMSEC')  
    plt.bar(ind+(14*width), MUSAE, width, color='coral', label='MUSAE') 
    plt.bar(ind+(15*width), PRUNE, width, color='maroon', label='PRUNE')
    plt.bar(ind+(16*width), ROLE2VEC, width, color='darkblue', label='ROLE2VEC')  
    plt.bar(ind+(17*width), SINE, width, color='darkorange', label='SINE') 
    
    plt.ylabel('Outliers Precision')
    plt.xticks(ind+width/2, algo)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=8, mode="expand", borderaxespad=0.)
    plt.show()
#anomaly_bar()


def ARI_clustering():
    df=pd.DataFrame({'x': range(38,43), 'NODE2VEC': (0.7092,0.7417,0.7770,0.8373,0.8867), 'BANE': (0.6598,0.6782,0.7159,0.7455,0.7720), 
    'WALKLETS': (0.6776,0.7034,0.7351,0.7520,0.7993), 'SINE': (0.6604,0.6697,0.6872,0.7239,0.7401), 'PRUNE': (0.5923,0.6335,0.6656,0.6748,0.6914), 
    'SPLITTER': (0.6588,0.6742,0.6948,0.7241,0.7486), 'DEEPWALK': (0.6407,0.7015,0.7038,0.7369,0.7743), 'GRAREP': (0.6404,0.6758,0.6823,0.7112,0.7391), 
    'BOOSTEDNE': (0.6650,0.6971,0.7111,0.7424,0.7748), 'HIVEC': (0.6692,0.7059,0.7207,0.7441,0.7817), 'ROLE2VEC': (0.6562,0.6975,0.7103,0.7320,0.7651), 
    'DANE': (0.6581,0.6634,0.6873,0.7262,0.7391), 'DIFF2VEC': (0.6876,0.7043,0.7448,0.7542,0.7869), 'MUSAE': (0.7056,0.7377,0.7761,0.7962,0.8197), 
    'GRAPHWAVE': (0.6823,0.7027,0.7220,0.7582,0.7793), 'DANMF': (0.6912,0.7323,0.7604,0.7914,0.8021), 'GEMSEC': (0.6720,0.6998,0.7125,0.7707,0.7873), 
    'FSCNMF': (0.6833,0.7306,0.7824,0.8428,0.8979), 'TADW': (0.6542,0.6923,0.7251,0.7631,0.7821)})
    plt.plot('x', 'DANE', data=df, marker='s', color='brown', linewidth=1, label="DANE")
    plt.plot('x', 'BANE', data=df, marker='s', color='green', linewidth=1, label="BANE")
    plt.plot('x', 'DEEPWALK', data=df, marker='s', color='blue', linewidth=1, label="DEEPWALK")
    plt.plot('x', 'FSCNMF', data=df, marker='s', color='chocolate', linewidth=1, label="FSCNMF")
    plt.plot('x', 'GRAPHWAVE', data=df, marker='s', color='black', linewidth=1, label="GRAPHWAVE")
    plt.plot('x', 'GRAREP', data=df, marker='s', color='springgreen', linewidth=1, label="GRAREP")
    plt.plot('x', 'NODE2VEC', data=df, marker='s', color='magenta', linewidth=1, label="NODE2VEC")
    plt.plot('x', 'TADW', data=df, marker='s', color='grey', linewidth=1, label="TADW")
    plt.plot('x', 'HIVEC', data=df, marker='s', color='orange', linewidth=1, label="HIVEC")
    plt.plot('x', 'WALKLETS', data=df, marker='s', color='pink', linewidth=1, label="WALKLETS")
    plt.plot('x', 'BOOSTEDNE', data=df, marker='s', color='red', linewidth=1, label="BOOSTEDNE")
    plt.plot('x', 'DANMF', data=df, marker='s', color='darkblue', linewidth=1, label="DANMF")
    plt.plot('x', 'DIFF2VEC', data=df, marker='s', color='violet', linewidth=1, label="DIFF2VEC")
    plt.plot('x', 'GEMSEC', data=df, marker='s', color='olive', linewidth=1, label="GEMSEC")
    plt.plot('x', 'MUSAE', data=df, marker='s', color='darkorange', linewidth=1, label="MUSAE")
    plt.plot('x', 'PRUNE', data=df, marker='s', color='gold', linewidth=1, label="PRUNE")
    plt.plot('x', 'ROLE2VEC', data=df, marker='s', color='maroon', linewidth=1, label="ROLE2VEC")
    plt.plot('x', 'SINE', data=df, marker='s', color='purple', linewidth=1, label="SINE")
    plt.plot('x', 'SPLITTER', data=df, marker='s', color='coral', linewidth=1, label="SPLITTER")
    
    plt.ylabel('ARI')
    plt.xlabel('Cluster Blocks')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=8, mode="expand", borderaxespad=0.)
    
    plt.show()
ARI_clustering() 

#Values from each model's runtime and space complexity
def time_space():
    df=pd.DataFrame({'model': ('DANE','BANE','DEEPWALK','FSCNMF','GRAPHWAVE','GRAREP','NODE2VEC','TADW','HIVEC','WALKLETS','BOOSTEDNE',
                               'DANMF','DIFF2VEC','GEMSEC','MUSAE','PRUNE','ROLE2VEC','SINE','SPLITTER', 'LINE'), 
                    'time': (0.8, 0.1, 0.5, 0.2, 3.4, 0.8, 0.2, 2.6, 0.3, 0.9, 1.0, 0.4, 0.2, 0.4, 8.9, 0.5, 9.1, 9.8, 0.4, 1.2),
                    'space': (0.2, 0.15, 0.15, 0.2, 0.2, 7.8, 0.1, 0.15, 0.2, 0.9, 0.1, 0.3, 1.4, 0.2, 0.9, 0.1, 1.0, 1.0, 0.3, 0.7)
                    })
    list_time = list(df['time'])
    list_space = list(df['space'])
    plt.figure(figsize=(10, 10), dpi=65)
    for i in range(len(list_time)):
        plt.plot('model', 'time', data=df, marker='s', color='blue', linewidth=1, linestyle='dashed', label="Training Run Time (seconds)")
        plt.annotate(list_time[i], xy = (i, list_time[i]), xytext = (0, 0), textcoords='offset points', ha='right', va='bottom')
        plt.plot('model', 'space', data=df, marker='o', color='black', linewidth=1, label="Training Memory profiler (MiB)")
        plt.annotate(list_space[i], xy = (i, list_space[i]), xytext = (0, 0), textcoords='offset points', ha='right', va='bottom')
    
    plt.ylabel('Training Time and Memory - 10^3')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, borderaxespad=0.)
    plt.show()
#time_space()

avg_clustering_coeff = (28.5,26.8,30.8,27.3,25.2,36.0,22.9,23.3,35.6,36.2,10.2,26.1,21.3,29.1,22.7,37.9,34.3,23.3,31.1)
predicted_clusters = (37,33,44,36,28,41,27,31,41,41,12,32,27,35,27,43,38,29,45)

ind = np.arange(len(predicted_clusters))
width = 0.20

fig, ax = plt.subplots(figsize=(10, 10), dpi=70)
rects1 = ax.bar(ind - width/2, avg_clustering_coeff, width, label='% Average Clustering Coefficient')
rects2 = ax.bar(ind + width/2, predicted_clusters, width, label='Predicted Number of Clusters')

ax.set_title('Average Clustering Coefficient and Predicted Clusters for each Model. \n Dataset Ground-truth: Cluster Size = 42 | Average Cluster Coefficient = 39.94%')
ax.set_xticks(ind)
ax.set_xticklabels(('BANE', 'BOOSTEDNE', 'DANMF', 'DEEPWALK', 'DIFF2VEC', 'FSCNMF', 'GEMSEC', 'GRAPHWAVE', 'MUSAE', 'NODE2VEC',
                     'PRUNE', 'ROLE2VEC', 'SINE', 'SPLITTER', 'TADW', 'HIVEC', 'WALKLETS', 'GRAREP', 'DANE'))

ax.legend()


def autolabel(rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")

fig.tight_layout()

plt.show()