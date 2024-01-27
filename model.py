from sklearn.linear_model import LogisticRegression

# This class is ment to handle the model options
class logistic_regression():

    def __init__(self):
        self.model = LogisticRegression()
    
    def train_model(self, X_train, y_train):
        # Training the Logistic Regression model
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        # Predictions
        y_pred = self.model.predict(X_test)
        
        return y_pred