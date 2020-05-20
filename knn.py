import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class KNN:
    def __init__(self,nb_column = None,columns = None):
        assert(not (nb_column == None and columns == None)),'You need to specify either nb_column or columns'
        # By default the column names will c0, c1, c2...
        if columns == None:
            columns = ['c' + str(i) for i in range(nb_column - 1)]
            columns.append('label')
            columns.append('Distance')
        self.df = pd.DataFrame(columns = columns)


    def get_mat_conf(self,filenamme = None,df_training = None):
        assert(not (filename == None and df_training == None)),'You need so specify either filename or df_training'
        if df_training == None:
            df_training = pd.read_csv(filename,names = self.df.columns)
        
        
       
        
    def load_data(self, filename, sep = ','):
        if self.is_valid_file(filename,sep = sep):
            temp_df = pd.read_csv(filename,names = self.df.columns, sep = sep)
            self.df = self.df.append(temp_df)
            return True
        return False

    def clear_data(self):
        self.df = self.df.iloc[0:0]
        
    def knn_alg(self,point,k):
        for i in range(len(self.df.index)):
            distance = self.get_eucl_dist(point,self.df.iloc[i].values.tolist()[:-1])
            self.df.at[i,'Distance'] = distance

        # Sort self.df by Distance
        self.df.sort_values(by = ['Distance'])
        
        # Get the label with highest frequence between the first k columns of self.df
        best_label = self.df.head(k)['label'].mode().iloc[0]
        return best_label

    def get_eucl_dist(self,p1,p2):
        distance = 0
        for i in range(len(p1)):
            distance += (p1[i]-p2[i])**2
        return math.sqrt(distance)

    def clear_distance(self):
        self.df['Distance'] = None

    def is_valid_file(self,filename, sep = ','):
        with open(filename) as file:
            for line in file.readlines():
                if len(line.split(';'))!=len(self.df.columns) - 1:
                    return False 
        return True

    def is_valid_df(self,df,has_label = True):



    def display_df(self):
        print(self.df)
        print(self.df.describe(),end="\n\n")
        print(self.df.dtypes)

    def display_pca2(self):
        # Code from : https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
        features = self.df.columns[:-2]
        x = self.df.loc[:, features].values
        x = StandardScaler().fit_transform(x)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, self.df[['label']]], axis = 1)

        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        labels = self.df.label.unique()
        for label,i in zip(labels,range(len(self.df.index))):
            indicesToKeep = finalDf['label'] == label
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c = 'C' + str(i)
                       , s = 50)
            i+=1
        ax.legend(labels)
        ax.grid()
        print(pca.explained_variance_ratio_)                
        plt.show()

            

test = KNN(5)
test.load_data("data.csv",sep = ';')
test.display_pca2()
