import numpy as np

### All code is Same except passing the prediction logits thorugh sigmoid function the doing prediction based on threshold...
class LogisticRegressionScratch:
    def __init__(self,lr,epoch):
        self.lr=lr
        self.epoch=epoch
        self.w=None
        self.b=None
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def fit(self,X,y):
        m,n=X.shape

        self.w=np.zeros(n)
        self.b=0

        for _ in range(self.epoch):
            z=np.dot(X,self.w)+self.b
            y_pred=self.sigmoid(z)

            dw=1/m*np.dot(X.T,(y_pred-y))
            db=1/m*np.sum(y_pred-y)

            self.w-=self.lr*dw
            self.b-=self.lr*db
        
    def predic_prob(self,X):
        z=np.dot(X,self.w)+self.b
        return self.sigmoid(z)
    
    def prediction(self,X,threshold):
        pred=self.predic_prob(X)
        return(pred>=threshold).astype(int)
    
    def get_params(self):
        return{"weights":self.w,"bias":self.b}
    
if __name__=="__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1])  # class labels

    model=LogisticRegressionScratch(0.01,2000)
    model.fit(X,y)
    probs=model.predic_prob(X)
    prediction=model.prediction(X,0.5)

    print("Predicted Probabilities:", probs)
    print("Predicted Classes:", prediction)
    print("Model Parameters:", model.get_params())

