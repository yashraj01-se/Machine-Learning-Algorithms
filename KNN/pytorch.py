###### So in PyTorch, KNN is implemented using tensor operations, not neural network layers.
import torch
from collections import Counter

class KNNClassifier:
    def __init__(self,k=3):
        self.k=k
        self.x_train=None
        self.y_train=None

    def fit(self,X,y):
        self.x_train=X
        self.y_train=y
    
    def predict(self,X):
        predictions=[]

        for x in X:
            distances = torch.sqrt(torch.sum((self.x_train - x)**2, dim=1)) #Finding the Upcoming tensor distance with all other training tensors...
            k_indices=torch.argsort(distances)[:self.k] # finding top k tensors with least distance...
            k_nearest_neighbour=self.y_train[k_indices] # finding its labels...
            most_common=Counter(k_nearest_neighbour.tolist()).most_common(1) # finding majority label...
            predictions.append(most_common[0][0]) #appending the prediction...

        return torch.tensor(predictions)
    
if __name__=="__main__":
    X_train = torch.tensor([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 3.0],
        [6.0, 5.0],
        [7.0, 7.0],
        [8.0, 6.0]
    ])

    y_train = torch.tensor([0, 0, 0, 1, 1, 1])

    X_test = torch.tensor([
        [2.0, 2.0],
        [7.0, 6.0]
    ])

    knn=KNNClassifier()
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)
    print("Prediction:",pred)
    
