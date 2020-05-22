import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math
import matplotlib.pyplot as plt

class KNN:

    def __init__(self,nb_column = None,columns = None):
        assert(not (nb_column == None and columns == None)),'You need to specify either nb_column or columns'
        # By default the column names will c0, c1, c2...
        if columns == None:
            columns = ['c' + str(i) for i in range(nb_column - 1)]
            columns.append('label')
            columns.append('Distance')
        self.df_training = pd.DataFrame(columns = columns)
        self.df_test = pd.DataFrame(columns = columns[:-1])
        self.scaler = MinMaxScaler()

    def scale(self,df):
        features = self.df_training.columns[:-2]
        temp_df = pd.concat([self.df_training.loc[:,features],self.df_test.loc[:,features]],axis = 0)
        self.scaler.fit(temp_df)
        return self.scaler.transform(df.values)

    def get_mat_conf(self, k = 7):
        label_unique = self.df_training.label.unique()
        mat_conf = np.array([[0 for i in range(len(label_unique))] for j in range(len(label_unique))])
        label_index = {label_unique[i] : i for i in range(len(label_unique))}
        
        # Scale data
        training_points = self.scale(self.df_training.loc[:,self.df_training.columns[:-2]])
        actual_labels = self.df_test.label.tolist()
        test_points = self.scale(self.df_test.loc[:,self.df_test.columns[:-1]])
        predicted_labels = []

        for i in range(len(self.df_test.index)):
            print(i,'/',len(self.df_test.index),end = '\r',flush = True)
            predicted_label = self.lm_dw_knn_alg(training_points,test_points[i],k)
            predicted_labels.append(predicted_label)
            mat_conf[label_index[actual_labels[i]], label_index[predicted_label]] += 1

        return mat_conf,predicted_labels

    def display_knn_stat(self,k = 7):
        mat_conf,predicted_labels = self.get_mat_conf(k)
        print(mat_conf)

        precisions = self.get_precisions(mat_conf)
        total_precision = self.get_total_precision(mat_conf)

        df_predicted = self.df_test.copy()
        df_predicted.drop('label',1, inplace = True)
        df_predicted['label'] = predicted_labels

        fig1 = self.get_pca2_fig(self.df_test,title = 'Actual label')
        fig2 = self.get_pca2_fig(df_predicted,2,title = 'Predicted label')        
        fig1.savefig('actual_label_pca.png')
        fig2.savefig('predicted_label_pca.png')

         
        for key,value in precisions.items():
            print('Label',key,'precision :',value*100,'%')
        print('Global precision : ',total_precision*100)
        plt.show()

    def get_total_precision(self,mat_conf):
        labels = self.df_training.label.unique()
        good_prediction = 0
        for i in range(len(labels)):
            good_prediction += mat_conf[i,i]
        total_precision = good_prediction/np.sum(mat_conf)
        return total_precision

    def get_precisions(self,mat_conf):
        labels = self.df_training.label.unique()
        eff_label = np.sum(mat_conf,axis = 0)
        precision = [0 for i in range(len(labels))]
        for i in range(len(labels)):
            precision[i] = mat_conf[i, i]/eff_label[i]
        precision = dict(zip(labels,precision))
        return precision

    def test_k(self,k_test = range(1,20)):
        precisions = []
        for k in k_test:
            print('Test k =',k)
            mat_conf,predicted_labels = self.get_mat_conf(k)
            precisions.append(self.get_total_precision(mat_conf))

        for value,key in dict(zip(k_test,precisions)).items():
            print('k =',value,'precision',key*100,'%')
        return precisions

    def get_mean_test_k(self,rep = 5):
        precision = []
        for i in range(rep):
            self.mix_data()
            precision.append(self.test_k(k_test = range(5,13)))

        mean_precision = [0 for i in range(5,13)]
        for i in range(len(precision[0])):
            mean_precision[i] = sum([p[i] for p in precision])/len(precision[0])

        for k in range(5,13):
            print('k :',k,'mean precision',mean_precision[k-5])

        plt.plot(range(5,13),mean_precision)
        plt.savefig('mean_test_k')
        plt.show()

    def load_csv_training(self, filename, sep = ','):
        if self.is_valid_file(filename,sep = sep):
            temp_df = pd.read_csv(filename,names = self.df_training.columns, sep = sep)
            self.df_training = self.df_training.append(temp_df)
            self.df_training.dropna(how = 'any',inplace = True,subset = self.df_training.columns[:-2])
            return True
        return False

    def load_csv_test(self,filename,sep = ','):
        if self.is_valid_file(filename,sep = sep,training = False):
            temp_df = pd.read_csv(filename,names = self.df_test.columns, sep = sep)
            self.df_test = self.df_test.append(temp_df)
            self.df_test.dropna(how = 'any',inplace = True,subset = self.df_test.columns)
            return True
        return False

    def clear_data_training(self):
        self.df_training = self.df_training.iloc[0:0]

    def clear_data_test(self):
        self.df_test = self.df_test.iloc[0:0]

    def mix_data(self,ratio_training = 0.8):
       features = self.df_training.columns[:-1]
       temp_df = pd.concat([self.df_training.loc[:,features],self.df_test.loc[:,features]],axis = 0)
       temp_df.reset_index(inplace = True,drop = True)
       print(len(temp_df.index))
       self.df_training = temp_df.sample(int(ratio_training*len(temp_df.index))).copy()
       temp_df = temp_df.drop(self.df_training.index)
       self.df_training['Distance'] = np.NaN
       self.df_test = temp_df.copy()
       self.df_test.reset_index(inplace = True,drop = True)
       self.df_training.reset_index(inplace = True,drop = True)
   
    def knn_alg(self,training_points,point,k):
        # Compute the distance
        for i in range(np.size(training_points,0)):
            distance = np.linalg.norm(point-training_points[i])
            self.df_training.at[i,'Distance'] = distance

        # Sort self.df_training by Distance
        temp_df = self.df_training.sort_values(by = ['Distance'])
            
        # Get the label with highest frequence between the first k columns of self.df_training
        best_label = temp_df.head(k)['label'].mode().iloc[0]
        return best_label

    def lm_dw_knn_alg(self,training_points,point,k):
        # Compute the distance
        for i in range(np.size(training_points,0)):
            distance = np.linalg.norm(point-training_points[i])
            self.df_training.at[i,'Distance'] = distance

        # Sort self.df_training by Distance
        temp_df = self.df_training.sort_values(by = ['Distance'])


        neighbors = pd.DataFrame(columns = ['label','Distance'])

        # Get the k nearest neighbors from each class
        for label in self.df_training.label.unique():
            neighbors = neighbors.append(temp_df.loc[temp_df['label'] == label].loc[:,['label','Distance']].head(k))

        sum_weights = {label : 0 for label in self.df_training.label.unique()}
        for i in range(len(neighbors.index)):
            sum_weights[neighbors.iloc[i].label] += 1 - neighbors.iloc[i].Distance

        best_value = -1
        best_label = None
        for key,value in sum_weights.items():
            value = value* len(self.df_test.index)/k
            if value>best_value:
                best_value = value
                best_label = key

        return best_label
  
    def get_eucl_dist(self,p1,p2):
        distance = 0
        for i in range(len(p1)):
            distance += (p1[i]-p2[i])**2
        return math.sqrt(distance)

    def clear_distance(self):
        self.df_training['Distance'] = None

    def is_valid_file(self,filename, sep = ',',training = True):
        with open(filename) as file:
            for line in file.readlines():
                if len(line.split(';'))!=len(self.df_training.columns) - 1:
                    return False 
        return True

    def get_pca2_fig(self,df,figure = 1,title = "PCA"):
        # Code from : https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
        features = self.df_training.columns[:-2]
        x = df.loc[:, features]
        x = self.scale(x)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, df[['label']]], axis = 1)

        fig = plt.figure(figure,figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title(title, fontsize = 20)
        labels = df.label.unique()
        labels.sort()
        for label,i in zip(labels,range(len(df.index))):
            indicesToKeep = finalDf['label'] == label
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c = 'C' + str(i)
                       , s = 50)
            i+=1
        ax.legend(labels)
        ax.grid()
        return fig

    def display_pca2_training(self):
        fig = self.get_pca2_fig(self.df_training)
        fig.show()
        input()
        
    def display_pca2_test(self):
        fig = self.get_pca2_fig(self.df_test)
        fig.show()
        input()

    def save_labels(self,file_data,k = 5):
        global_precision = 0
        while global_precision<0.895:
            self.mix_data()
            mat_conf = self.get_mat_conf(k)[0]
            global_precision = self.get_total_precision(mat_conf)
            print(global_precision)

        label_unique = self.df_training.label.unique()
        
        self.df_test = pd.read_csv(file_data,names = self.df_training.columns[:-2],sep = ';')
        # Scale data
        training_points = self.scale(self.df_training.loc[:,self.df_training.columns[:-2]])
        test_points = self.scale(self.df_test.loc[:,self.df_test.columns])
        predicted_labels = []

        for i in range(len(self.df_test.index)):
            print(i,'/',len(self.df_test.index),end = '\r',flush = True)
            predicted_label = self.lm_dw_knn_alg(training_points,test_points[i],k)
            predicted_labels.append(predicted_label)

        with open('output_Alizadeh.txt','w') as file:
            for label in predicted_labels:
                file.write(label+'\n')



knn = KNN(5)
knn.load_csv_training('data.csv',sep = ';')
knn.load_csv_test('preTest.csv', sep = ';')
knn.save_labels('finalTest.csv')
