from utilities import convert_to_vibration, estimate_fundFreq, filterBank, PowerModule, get_mean_std
from query_fn import query_it, query_betn

import yaml
import argparse
import pandas as pd
import numpy as np

def calibrate(vib_data_pd):
    vib_obj_data=convert_to_vibration(vib_data_pd)  ### vib_data_pd ===> pandas dataframe with time and vibration
    
    fundFreq=estimate_fundFreq(vib_data_pd)
    
    args={"fundFreq": fundFreq, 
      "fs": 1000, 
      "mean": [0, 0, 0, 0, 0, 0, 0],
      "std": [1, 1, 1, 1, 1, 1, 1]
     }
    
    fb=filterBank.from_dict({"fundFreq": fundFreq, "fs": 1000})         ###Creation of filter bank
    
    filtered_data=[fb.filter(x) for x in vib_obj_data]  ###Using the filterbank create filtered data list
    
    oPower=PowerModule.from_dict(args)   ### Create the PowerModuleused to convert filteres data into power data
    print(oPower.mean,oPower.std)
    power_O=[oPower.compute(x,normalize=False) for x in filtered_data]  ###Conversion of filtered data into power with 7 features each data point
    
    mean, std = get_mean_std(power_O)
    mean=list(mean)
    print(type(mean))
    std=list(std)
    
    return {"fundFreq": float(fundFreq), 
      "fs": 1000, 
      "mean": list(mean),
      "std": list(std)
     } 


def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-db",
        dest="DB",
        default=None,
        help="name of db containing the raw and processed data",
    )
    parser.add_argument(
        "-s",
        dest="SAVE_AT",
        default=None,
        help="folder to save the output args file at",
    )
    parser.add_argument(
        "-c",
        dest="CSV",
        default=None,
        help="location of csv file that is to be used to create the calibration file"
    )
    flags = vars(parser.parse_args())
    return flags

if __name__ == "__main__":
    flags = input_args()

    db_name = flags["DB"]
    csv = flags["CSV"]
    to_save_location = flags["SAVE_AT"]

    raw_vib=pd.read_csv(csv)
    args=calibrate(raw_vib)
    
    with open(to_save_location+"/args.yml", "w") as fid:
        yaml.dump(args, fid)


    print(args)