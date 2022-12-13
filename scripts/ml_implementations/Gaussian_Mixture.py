import argparse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import pickle


"""
python Gaussian_Mixture.py 
        -m "/home/spoudel/Thesis/Data/Lathrope/train_test/train/GMM_model.pkl" ==>>Location to the model
        -s "/home/spoudel/Thesis/Data/Lathrope/train_test/data_pass/l3_3781881_1212807"  ==>> path to the folder to save the data created
        -c "/home/spoudel/Thesis/Data/Lathrope/train_test/data_pass/l3_3781881_1212807/power_first_pass_3781881_1212807_105.csv"  ==>Power.csv file to train And/or predict 
        -sn "GMM_first_pass_3781881_1212807_105.csv"  ==> Name of the Csv file after prediction

"""

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
        help="location of csv file that is to b used to train the model"
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

# def cluster_to_prediction(prediction,model):
#     """Order each cluster by sum"""
#     N = model.means_.copy()        
#     weights=np.array([1,0.5,0.5,0.5,0.25,0.25,0.25])
    
#     weighted_N=N * weights
#     sum_N = np.sum(weighted_N, axis=1)
#     print(f"1. {sum_N}")
#     sorted_N = np.argsort(sum_N)
#     argdash=list(sorted_N)
#     print(argdash)
#     return [argdash.index(x) for x in prediction]

if __name__=="__main__":
    flags=input_args()

    csv=flags["CSV"]
    save_at=flags["SAVE_AT"]
    model_l=flags["MODEL"]
    save_name=flags["SAVE_NAME"]
    model_name=flags["MODEL_NAME"]

    training_data=pd.read_csv(csv)
    x = training_data.to_dict("records")
    x = [[d["p0"], d["p1"], d["p2"], d["p3"], d["p4"], d["p5"], d["p6"]] for d in x]
    np_array_train = np.array(x)
    
    if model_l:
        with open(model_l, 'rb') as f:
            model = pickle.load(f)
    else:
        model=GaussianMixture(n_components=5,covariance_type='full',random_state=42)
        model.fit(np_array_train)
        with open(save_at+'/'+model_name,'wb') as f:
            pickle.dump(model,f)

    
    GMM_prediction=model.predict(np_array_train)
    # GMM_prediction=cluster_to_prediction(GMM_prediction,model)
    
    training_data["GMM"]=GMM_prediction

    training_data.to_csv(save_at+"/"+save_name)

    # print(model.means_)
  
   