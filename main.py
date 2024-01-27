from fastapi import FastAPI

from preprocessing import data_handler
from model import logistic_regression
from postprocessing import serialize_ndarray


app = FastAPI()

# Instantiates model object
model = logistic_regression()

@app.get("/")
async def root():
    
    # Instantiates data handler object
    input_data = data_handler()
    # Loads data from dataset (.csv file)
    input_data.load_data()
    # Handles null values
    imputed_data = input_data.handle_null_values()
    # splits data into features and targets
    input_data.features_target_data_split(imputed_data)
    # Splits data into train and test 
    input_data.train_test_data_split(input_data.X, input_data.y)
    
    # Trains model on current input data
    model.train_model(input_data.X_train, input_data.y_train)
    # Computes inference on current input data
    y_pred = model.predict(input_data.X_test)

    print(y_pred)
    # Serialize output to be sent as quiery response
    serialized_output = serialize_ndarray(y_pred)

    return {"message": serialized_output}
