import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle

def input_args():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        dest="SAVE_AT",
        help="folder to save the output dataframes at",
    )
    parser.add_argument(
        "-c",
        dest="CSV",
        help="location of POWER power_final_pass_3781881_1212807_114.csv file that is to be clustered or used to train the model"
    )
    parser.add_argument(
        "-sn",
        dest="SAVE_NAME",
        help="name of the predicted dataframe"
    )
    parser.add_argument(
        "-m",
        dest="MODEL",
        default=None,
        help="location of the trained model"
    )
    parser.add_argument(
        "-mn",
        dest="MODEL_NAME",
        default=None,
        help="Name to save the new created model"
    )
    flags = vars(parser.parse_args())
    return flags

def sort_clusters_by_total_power(model):
    """Order each cluster by sum"""
    N = model.cluster_centers_.copy()
    sum_N = np.sum(N, axis=1)
    logging.debug(f"1. {sum_N}")
    sorted_N = np.argsort(sum_N)
    logging.debug(f"2. {sorted_N}")
    model.cluster_centers_ = N[sorted_N, :]
    return model

def sort_clusters(model):
    """Sort clusters by power bands."""
    N = model.cluster_centers_.copy()
    for i in range(model.n_features_in_):
        idx = np.argsort(N[:, i])
        idx = N[:, i].argsort()
        model.cluster_centers_[:, i] = N[idx, i]
    print(model.cluster_centers_)
    return model

if __name__=="__main__":
    flags=input_args()

    csv=flags["CSV"]
    save_at=flags["SAVE_AT"]
    model_l=flags["MODEL"]
    save_name=flags["SAVE_NAME"]
    model_name=flags["MODEL_NAME"]

    training_data=pd.read_csv(csv)
    # reshape Training Data
    print(training_data.head())
    x = training_data.to_dict("records")
    x = [[d["p0"], d["p1"], d["p2"], d["p3"], d["p4"], d["p5"], d["p6"]] for d in x]
    np_array_train = np.array(x)
    
    if model_l:
        with open(model_l, 'rb') as f:
            model = pickle.load(f)
    else:
        model=KMeans(n_clusters=6)
        model.fit(np_array_train)
        model=sort_clusters(model)
        with open(save_at+'/'+model_name,'wb') as f:
            pickle.dump(model,f)

    
    K_Means_prediction=model.predict(np_array_train)
    
    training_data["K_Means"]=K_Means_prediction

    training_data.to_csv(save_at+"/"+save_name)

