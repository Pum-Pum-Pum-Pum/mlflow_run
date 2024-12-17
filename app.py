# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_squared_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Lets read the wine quality data
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )   
    
    try:
        data = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(
            "Unable to download wine data, looks like URL is wrong or internet issue", e
        )
    
    # Spliting into train and test as default 0.75, 0.25
    
    train, test = train_test_split(data)
    
    # The predicted column is "quality" which is in the scale from (3,9)
    train_x = train.drop(["quality"], axis = 1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,random_state=42)
        lr.fit(train_x, train_y)
        
        predicted_qualites = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualites)
        
        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)
        
        ## For Remote server only(DAGShub) 
        ## Needs to be updated
        ##remote_server_uri="https://dagshub.com/krishnaik06/mlflowexperiments.mlflow" ## Update
        ## mlflow.set_tracking_uri(remote_server_uri)
        ## tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        input_example = train_x.iloc[:1]
        
        
        # Set MLflow to log locally
        mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel",signature=signature, input_example = input_example
            )
        
        ## Model registry does not work with file store
        ## if tracking_url_type_store != "file":
        ##     mlflow.sklearn.log_model(
        ##         lr, "model", registered_model_name="ElasticnetWineModel",signature=signature
        ##     )
        ## else:
        ##     mlflow.sklearn.log_model(lr, "model")
        