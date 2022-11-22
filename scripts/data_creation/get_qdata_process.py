from influxdb import InfluxDBClient
import pandas as pd
import dataclasses
import numpy as np
from scipy import signal
import argparse
import yaml

from utilities import convert_to_vibration, estimate_fundFreq, filterBank, PowerModule, get_mean_std
from query_fn import query_it, query_betn


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
        help="folder to save the output dataframes at",
    )
    parser.add_argument(
        "-c",
        dest="CSV",
        default=None,
        help="location of csv file that is to b used to create the calibration file"
    )
    parser.add_argument(
        "-arg",
        dest="ARGS",
        default=None,
        help="location of csv file that is to b used to create the calibration file"
    )
    parser.add_argument(
        "-sn",
        dest="SAVE_N",
        default=None,
        help="location of csv file that is to b used to create the calibration file"
    )
    flags = vars(parser.parse_args())
    return flags


def convert_vib_df_to_power_list(vib_data_pd,args,normalize=True):
    """
    Input:
    vib_data_pd= pandas dataframe with time and vibration
    args=arguments to create the filterbank and powerobject
    normalize=weather to get the normalized value or non-normalized values
    
    Output:
    list of power object
    """
    
    vib_obj_data=convert_to_vibration(vib_data_pd)  ### vib_data_pd ===> pandas dataframe with time and vibration
   
    fb=filterBank.from_dict({"fundFreq": args["fundFreq"], "fs": args["fs"]})         ###Creation of filter bank
    
    filtered_data=[fb.filter(x) for x in vib_obj_data]  ###Using the filterbank create filtered data list

    oPower=PowerModule.from_dict(args)   ### Create the PowerModule for the 
    print(args,"\n",oPower.mean,"\n",oPower.std)
    power0=[oPower.compute(x,normalize=normalize) for x in filtered_data]
    return power0

if __name__ == "__main__":
    flags = input_args()

    db_name = flags["DB"]
    to_save_location = flags["SAVE_AT"]
    to_save_name=flags["SAVE_N"]
    csv = flags["CSV"]
    aarg_l= flags["ARGS"]


    with open(aarg_l, "r") as fid:
        aarg = yaml.safe_load(fid)

        print(aarg)

    data_to_convert=pd.read_csv(csv)

    power_o=convert_vib_df_to_power_list(data_to_convert, aarg, normalize=True)
    power_o1=convert_vib_df_to_power_list(data_to_convert, aarg, normalize=False)

    if power_o==power_o1:
        print("normalize may not be working")

    power_df=pd.DataFrame([s.to_dict() for s in power_o])
    power_df.to_csv(to_save_location+"/"+to_save_name)










