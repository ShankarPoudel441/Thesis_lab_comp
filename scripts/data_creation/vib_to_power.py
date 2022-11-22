from utilities import convert_to_vibration, estimate_fundFreq, filterBank, PowerModule

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
    power0=[oPower.compute(x) for x in filtered_data]
    return power0

