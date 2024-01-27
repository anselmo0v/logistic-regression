import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# This functions could be used to retreive data from
# a data storage in specific use cases
def download_input_data():

    return "dataset.csv"

# This class is ment to handle data to be fed to the
# model
class data_handler():

    def __init__(self, data_file="dataset.csv"):
        self.data_file = data_file
        self.imputer = SimpleImputer(fill_value=0)
        
    def load_data(self):
        # Load dataset
        self.df = pd.read_csv(self.data_file)
    
    def handle_null_values(self):
        # Handling null values
        df_imputed = self.df.copy()
        df_imputed[['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Number_of_Open_Accounts']] = self.imputer.fit_transform(
            self.df[['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Number_of_Open_Accounts']]
            )
        return df_imputed
    
    def features_target_data_split(self, df_imputed):
        # Splitting the data into features and target
        self.X = df_imputed.drop('Loan_Approval', axis=1)
        self.y = df_imputed['Loan_Approval']

    def train_test_data_split(self, X, y):
        # Splitting the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.01, random_state=42)