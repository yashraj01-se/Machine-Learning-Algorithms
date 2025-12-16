import numpy as np
from collections import Counter

class KNN:
    def __init__(self,k=3):
        self.k=k
        self.x_train=None
        self.y_train=None
    
    def fit(self,X,y):
        self.x_train=X
        self.y_train=y
    
    def ecladian(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self,X):
        prediction=[]

        for x in X:
            distances=[
                self.ecladian(x,x_train)
                for x_train in  self.x_train
            ]

            k_indices=np.argsort(distances)[:self.k] # finding top k Indices
            k_nearest_neighbour=[self.y_train[i] for i in k_indices] # finding labels
            most_common=Counter(k_nearest_neighbour).most_common(1) # majority label
            prediction.append(most_common[0][0]) # appending that...

        return np.array(prediction)
    
if __name__=="__main__":
     # Simple 2D dataset
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 7],
        [8, 6]
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [2, 2],
        [7, 6]
    ])

    knn=KNN(3)
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)
    print("Prediction:",pred)

