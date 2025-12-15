import numpy as np #For The Numerical Computations

class LinearRegressionfromScratch:
    def __init__(self,lr=0.01,epoch=20):
        self.lr=lr
        self.epoch=epoch
        self.w=None
        self.b=None

    def fit(self,X,y):
        m,n=X.shape #m=Number of samples #n=number of features

        self.w=np.zeros(n) #The number of weights must equal the number of input features.
        self.b=0

        for _ in range(self.epoch):
            y_pred=np.dot(X,self.w)+(self.b)

            dw=1/m*np.dot(X.T,(y_pred-y)) #Derivative of loss with respect to weights...
            db=1/m*np.sum(y_pred-y) #Derivative of loss with respect to bias...

            self.w-=self.lr*dw  #The loss function is implicitly used because we directly compute its gradients. Gradient descent only needs the derivatives of the loss, not the loss value itself.
            self.b-=self.lr*db

    def predict(self,X):
        return np.dot(X,self.w)+(self.b)

    def get_params(self):
        return {"weights":self.w,"bias":self.b}
    
if __name__=="__main__":
    #sample Data:
    X = np.array([[1], [2], [3], [4], [5]]) # m,n=(5,1)
    y = np.array([3, 5, 7, 9, 11])  

    model=LinearRegressionfromScratch(lr=0.01,epoch=2000)
    model.fit(X,y)

    preds_after=model.predict(X)
    print("Model Predictions after training:",preds_after)
    print("Model parameters:",model.get_params())


