import json

def serialize_ndarray(y_pred):
    return json.dumps(y_pred.tolist())