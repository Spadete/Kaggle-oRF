import numpy as np
from scipy.stats import mode
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Node():
    def __init__(self, 
                 feature_i_star=None, 
                 th_star=None, 
                 left=None, 
                 right=None, 
                 label=None):
        
        # for decision node
        self.feature_i_star = feature_i_star
        self.th_star = th_star
        self.left = left
        self.right = right
        
        # y_hat for leaf nodes
        self.label = label

class DecisionTreeClassifier():
    def __init__(self):
        self.root = None
        
    def build_tree(self, X, Y):

        if len(np.unique(Y)) == 1: #StopCriteria
            return Node(label=Y[0]) #retorna y_hat

        n, m = X.shape

        feature_i_star, th_star = self.get_best_split(X, Y) 

        if feature_i_star == -1:
             return Node(label=self.calculate_leaf_label(Y))

        #***Implemente*** o split dos dados para enviar para os filhos
        Xi_left = []
        Yi_left = []
        Xi_right = []
        Yi_right = []
        for i in range(n):
            if X[i][feature_i_star] > th_star:
                Xi_right.append(X[i, :])
                Yi_right.append(Y[i]) 
            else:
                Xi_left.append(X[i, :])
                Yi_left.append(Y[i])
        
        Xi_left = np.array(Xi_left) #transformando em arrays numpy pra aceitar o método shape durante a recursão
        Xi_right = np.array(Xi_right)
        
        if len(Xi_left) == 0 or len(Xi_right) == 0:
            return Node(label=self.calculate_leaf_label(Y))

        left_subtree = self.build_tree(Xi_left, Yi_left)
        right_subtree = self.build_tree(Xi_right, Yi_right)
        
        return Node(feature_i_star, th_star, 
                    left_subtree, right_subtree)
    
    def get_best_split(self, X, Y):
            i_star, th_star = -1, -1
            max_info_gain = -float("inf")
            
            m = X.shape[-1]
            n = X.shape[0]
            for feature_i in range(m):
                feature_values = X[:, feature_i] #salvando todos os valores de uma feature num array
                
                #***Implemente*** os valores de limiares
                thresholds = np.unique(feature_values)
                
                for th in thresholds:
                                          
                        #***Implemente*** o split dos dados
                        Xi_right = []
                        Yi_right = []
                        Xi_left = []
                        Yi_left = []
                        for i in range(n):
                            if X[i][feature_i] > th:
                                Xi_right.append(X[i, :])
                                Yi_right.append(Y[i])
                            
                            else:
                                Xi_left.append(X[i, :])
                                Yi_left.append(Y[i])
                                
                        Yi_left = np.array(Yi_left)
                        Yi_right = np.array(Yi_right)
                        if len(Xi_left)>0 and len(Xi_right)>0:
                            curr_info_gain = self.information_gain(Y, Yi_left, Yi_right)

                            if curr_info_gain>max_info_gain:
                                i_star = feature_i
                                th_star = th
                                max_info_gain = curr_info_gain
                            
            return i_star, th_star
    
    def entropy_calc(self, y):
         values, counts = np.unique(y, return_counts=True)
         p = counts / counts.sum()
         return entropy(p, base=2)

    def information_gain(self, parent, l_child, r_child):
           #***Implemente*** a função de ganho de informação 
        eta = len(l_child)/len(parent)
        return self.entropy_calc(parent) - (eta*self.entropy_calc(l_child) + (1-eta)*self.entropy_calc(r_child))
    
        
    def calculate_leaf_label(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)

    def predict(self, X):
        predictions = []
        for x in X:
            y_hat  = self.make_prediction(x, self.root)
            predictions.append(y_hat)

        return predictions

    def make_prediction(self, x, tree):
        
        if tree.label is not None: #Leaf
            return tree.label
        
        feature_val = x[tree.feature_i_star]
        if feature_val<=tree.th_star:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
class RF ():

    def __init__ (self, k=10, p=0.6, s=0.1):
        self.k = k   # number of trees
        self.p = p # Subsets of 10% data (bagging)
        self.s = s #Subsets of s% features (feature subsampling)
        self.trees = []
        self.features_list = []

    trees = []
    features_list = []

    #Training phase
    def fit(self, X_train, Y_train):
        for _ in range(k):
            # bagging
            idx = np.random.choice(len(X_train), int(p * len(X_train)), replace=True)
            
            # feature selection
            feat_idx = np.random.choice(m, int(m*s), replace=False)
            
            X_sub = X_train[idx][:, feat_idx]
            y_sub = Y_train[idx]
            
            tree = DecisionTreeClassifier()
            tree.fit(X_sub, y_sub)
            
            self.trees.append(tree)
            self.features_list.append(feat_idx)

    #Testing phase
    def predict(self, X_test):
        preds = []
        for tree, feat_idx in zip(self.trees, self.features_list):
                y_hat = tree.predict(X_test[:, feat_idx])
                preds.append(y_hat)

        preds = np.array(preds)
        final_pred = mode(preds, axis=0, keepdims=False).mode
        return final_pred



def main():

    #n, m = 10000, 256
    #n_classes = 4
    #X, y = make_classification(n_samples=n, n_features=m, n_classes=n_classes, n_informative=m//2)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    n_train, n_test = map(int, input().split())
    X_train = np.zeros((n_train, 2), dtype=float)
    Y_train = np.zeros((n_train, 1), dtype=int)
    X_test = np.zeros((n_test, 2), dtype=float)
    
    
    for i in range(n_train):
        tmp = np.array(list(map(float, input().split())))
        X_train[i] = tmp[:-1]
        Y_train[i] = int(tmp[-1])

    for i in range(n_test):
        X_test[i] = np.array(list(map(float, input().split())))

    
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    y_hat = model.predict(X_test)
    
    accuracy_score(y_test,y_hat)

    main()