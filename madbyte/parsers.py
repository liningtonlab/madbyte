import logging
import os

import pandas as pd

from madbyte.logging import get_logger

# from lxml import etree

# Define module logger as child to App logger
logger = get_logger("MADByTE.parsers")


def topspin_parser(name, input_dir, output_dir, tocsy_precision=3, hsqc_precision=3):
    """Load Topspin data from specified input directory
    to preprocessed state. Dumps temp files to output_dir

    Args:
        name (str): Extract name
        input_dir (str): Directory name of input data
        output_dir (str): Directory name for output data
    """
    data_dir = os.path.join(input_dir, name)
    hsqc_dir = None
    tocsy_dir = None

    # Iterate over all dirs in topspin data dir for sample
    # to find which contains the correct data for experiments needed
    for dir_ in os.listdir(data_dir):
        exp_dir = os.path.join(data_dir, dir_)
        # In case a file exits in the path
        if not os.path.isdir(exp_dir):
            logger.warn(f"{exp_dir} is not a directory - skipping")
            continue
        with open(os.path.join(exp_dir,"pulseprogram")) as f:
            content = f.readline().strip()
            if "hsqc" in content:
                hsqc_dir = exp_dir
            elif "dipsi" in content or "cosy" in content:
                tocsy_dir = exp_dir
            elif "mlev" in content:
                tocsy_dir = exp_dir
            # Skip first line
            else: 
                next(f)
                content = f.readline().strip()
                if "hsqc" in content:
                    hsqc_dir = exp_dir
                elif "dipsi" in content or "cosy" in content:
                    tocsy_dir = exp_dir
                elif "mlev" in content:
                    tocsy_dir = exp_dir

    logger.debug(f"{name} : HSQC - %s", hsqc_dir)
    logger.debug(f"{name} : TOCSY - %s", tocsy_dir)

    # Raise an exception if missing any of the required data directories
    if not all([hsqc_dir, tocsy_dir]):
        logger.error("Missing data directory for samples %s", name)
        raise Exception(f"NMR data missing in TopSpin data directories for {name}")

    ## HSQC IMPORT
    hsqc_data = pd.DataFrame(topspin_text_reader(hsqc_dir, "hsqc")).astype("float")\
                    .round(hsqc_precision).sort_values(by=["H_PPM"], ascending=True)
    hsqc_data.Identity = name
    hsqc_data.to_json(
        os.path.join(output_dir, f"{name}_HSQC_Preprocessed.json"),
        orient="records"
    )
    ## TOCSY IMPORT
    tocsy_data = pd.DataFrame(topspin_text_reader(tocsy_dir, "tocsy")).astype("float")\
                    .round(tocsy_precision).sort_values(by=["Ha"], ascending=True)
    tocsy_data.Identity = name
    tocsy_data.to_json(
        os.path.join(output_dir, f"{name}_TOCSY_Preprocessed.json"),
        orient="records"
    )

    return hsqc_data, tocsy_data


def acd_sim_parser(name, input_dir, output_dir, precision=3):
    """Load ACD labs simulated data from specifed input directory
    to preprocessed state

    Args:
        name (str): Extract name
        input_dir (str): Directory name of input data
        output_dir (str): Directory name for output data
    """
    ## HSQC IMPORT

    # Get datafile and set column names
    # use_cols filters only the wanted columns
    hsqc_data = pd.read_csv(
        os.path.join(input_dir, "Text_Files", f"{name}_HSQC.txt"),
        skiprows=27,
        delimiter=" ",
        names=["H_PPM","C_PPM","Intensity","Normalized_Int","Identity","MT3","MT4"],
        usecols=["H_PPM","C_PPM","Intensity","Identity"],
    )
    hsqc_data.Identity = name + "_Sim"  # Mark the ID as a simulated spectra
    hsqc_data = hsqc_data.round(precision)

    hsqc_data.to_json(
        os.path.join(output_dir, f"{name}_HSQC_Preprocessed.json"),
        orient="records"
    )
    logger.debug("Loaded HSQC data for %s", name)

    ## TOCSY IMPORT

    # Process the "tocsy" data. ACD does not simulate TOCSY experiments,
    # but the cosy works under the same principal.
    tocsy_data = pd.read_csv(
        os.path.join(input_dir, "Text_Files", f"{name}_COSY.txt"),
        skiprows=27,
        delimiter=" ",
        names=["Ha", "Hb", "Intensity", "MT1", "MT2", "MT3", "MT4"],
        usecols=["Ha", "Hb", "Intensity"],
    )
    tocsy_data = tocsy_data.round(precision)
    tocsy_data.to_json(
        os.path.join(output_dir, f"{name}_TOCSY_Preprocessed.json"),
        orient="records"
    )
    logger.debug("Loaded TOCSY data for %s", name)

    return hsqc_data, tocsy_data


def topspin_text_reader(directory, dtype):
    """Use `peak.txt` file to get 2D peak list for specified
    directory

    Args:
        directory (str): Directory to get data from
        dtype (str): One of `tocsy` or `hsqc`
    Raises:
        e: Catch all exceptions and log them then raise

    Returns:
        data (list): List of dictionaries containing peak data
    """
    if dtype not in ("tocsy", "hsqc"):
        logger.error("Datatype not understood in text reader")
        raise Exception

    def line_filter(line):
        line = line.strip()
        return bool(line) and not line.startswith("#")

    data = []
    try:
        with open(os.path.join(directory, "pdata", "1", "peak.txt")) as f:
            lines = list(filter(line_filter, f.readlines()))
    except Exception as e:
        logger.error("Unable to read file - %s", e)
        raise e

    # Iterate of Text data
    for l in lines:
        dat = l.strip().split()
        if dtype == "hsqc":
            data.append(
                {
                    "H_PPM": dat[3],
                    "C_PPM": dat[4],
                    "Intensity": dat[5],
                }
            )
        elif dtype == "tocsy":
            data.append(
                {
                    "Ha": dat[3],
                    "Hb": dat[4],
                    "Intensity": dat[5],
                }
            )

    return data


def mestrenova_parser(name, input_dir, output_dir, tocsy_precision=3, hsqc_precision=3):
    '''Load in Mestrenova data from input directory - from peak table output
    To retreive data for this parser, in mestrenova: File->Save As->select peak table output->rename to a csv'''
    data_dir = os.path.join(input_dir)
    hsqc_dir = None
    tocsy_dir = None

    # Iterate over all dirs in MestReNova data dir for sample
    for dir_ in os.listdir(data_dir):
        if dir_ == name:
            sample_dir = os.path.join(data_dir, dir_)
                # In case a file exits in the path
            if not os.path.isdir(sample_dir):
                # logger.warn(f"{sample_dir} is not a directory - skipping")
                continue
            for file in [f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir,f))]:
                if 'HSQC' in file:
                    hsqc_dir = sample_dir
                if 'TOCSY' or 'COSY' in file:
                    tocsy_dir  = sample_dir
            ### HSQC and TOCSY Importing from Mestrenova
            for i in os.listdir(sample_dir):
                if 'HSQC' in i:
                    file = i
                    open_file = open(os.path.join(sample_dir,file),'r')
                    lines = open_file.readlines()
                    Data_Lines = lines[1:]
                    Data_Dict_List = []
                    for line in Data_Lines:
                        Data_Dict= dict()
                        Values = line.split('\t')
                        if len(Values)>1:
                            Data_Dict['C_PPM']=Values[0]
                            Data_Dict['H_PPM']=Values[1]
                            Data_Dict['Intensity']=Values[2]
                            Data_Dict_List.append(Data_Dict)
                    data = pd.DataFrame(Data_Dict_List)
                    hsqc_data = data.copy().astype("float").round(hsqc_precision).sort_values(by=["H_PPM"], ascending=True)
                elif 'TOCSY' or 'COSY' in i:
                    file = i
                    open_file = open(os.path.join(sample_dir,file),'r')
                    lines = open_file.readlines()
                    Data_Lines = lines[1:]
                    Data_Dict_List = []
                    for line in Data_Lines:
                        Data_Dict= dict()
                        Values = line.split('\t')
                        if len(Values)>1:
                            Data_Dict['Ha']=Values[0]
                            Data_Dict['Hb']=Values[1]
                            Data_Dict['Intensity']=Values[2]
                            Data_Dict_List.append(Data_Dict)
                    data = pd.DataFrame(Data_Dict_List)
                    tocsy_data = data.copy().astype("float").round(tocsy_precision).sort_values(by=["Ha"], ascending=True)
        else:
            logger.error('There was an issue finding the file for %s. Check to ensure that the name of the file and the name of the directory follow the correct conventions.',name)
    if not all([hsqc_dir, tocsy_dir]):
        logger.error("Missing data directory for samples %s", name)
        raise Exception(f"NMR data missing in data directories for {name}")
    ### HSQC Import finishing steps
    hsqc_data.Identity = name
    hsqc_data.to_json(
        os.path.join(output_dir, f"{name}_HSQC_Preprocessed.json"),
        orient="records"
    )
    ### TOCSY IMPORT finishing steps
    tocsy_data.Identity = name
    tocsy_data.to_json(
        os.path.join(output_dir, f"{name}_TOCSY_Preprocessed.json"),
        orient="records"
    )

    return hsqc_data, tocsy_data

def csv_parser(name, input_dir, output_dir, tocsy_precision=3, hsqc_precision=3):
    '''
    Allows for data to be put in from ambiguous csv files. 
    Headers MUST BE:
        H_PPM,C_PPM,Identity for HSQC data
        Ha,Hb,Identity for TOCSY data

    inputs:
    name(str): The name of the compound to parse
    input_dir(path/string): path to the input data
    output_dir(path/string): path to the output directory
    tocsy_precision(int): number of decimals for tocsy consideration
    hsqc_precision(int): number of decimals for hsqc consideration
    
    output: 
    hsqc_data(datframe): formatted HSQC data for MADByTE processing
    tocsy_data(dataframe): formatted TOCSY data for MADByTE processing
    JSON files for each are output in the output directory. 
    
    '''
    #Allows for manual input of data as CSV files # 
    Data_Path = os.path.join(input_dir,name)
    for data in os.listdir(Data_Path):
        if 'HSQC' in data: 
            hsqc_data = pd.read_csv(os.path.join(Data_Path,data))
            hsqc_data = hsqc_data.copy().astype("float").round(hsqc_precision).sort_values(by=["H_PPM"], ascending=True)
            hsqc_data.Identity = name
            hsqc_data.to_json(os.path.join(output_dir, f"{name}_HSQC_Preprocessed.json"),orient="records")
        if 'TOCSY' in data: 
            tocsy_data = pd.read_csv(os.path.join(Data_Path,data))
            tocsy_data = tocsy_data.copy().astype("float").round(tocsy_precision).sort_values(by=["Ha"], ascending=True)
            ### TOCSY IMPORT finishing steps
            tocsy_data.Identity = name
            tocsy_data.to_json(os.path.join(output_dir, f"{name}_TOCSY_Preprocessed.json"),orient="records")
        else:
            pass
    return hsqc_data,tocsy_data

def jeol_parser(name, input_dir, output_dir,tocsy_precision=3,hsqc_precision=3):
    '''Allows for import of JEOL Database formatted CSV files
    
    inputs:
    name(str): The name of the compound to parse
    input_dir(path/string): path to the input data
    output_dir(path/string): path to the output directory
    tocsy_precision(int): number of decimals for tocsy consideration
    hsqc_precision(int): number of decimals for hsqc consideration
    
    output: 
    hsqc_data(datframe): formatted HSQC data for MADByTE processing
    tocsy_data(dataframe): formatted TOCSY data for MADByTE processing
    JSON files for each are output in the output directory. 
    
    '''
    Data_Path = os.path.join(input_dir,name)
    for data in os.listdir(Data_Path):
        if 'HSQC' in data or 'hsqc' in data: 
            hsqc_data = pd.read_csv(os.path.join(Data_Path,data),skiprows=6) #Tested with database csv export option
            hsqc_data.X = hsqc_data.X.map(lambda x: x.rstrip('[ppm]'))
            hsqc_data.Y = hsqc_data.Y.map(lambda x: x.rstrip('[ppm]'))
            hsqc_data = hsqc_data.rename(columns={"X":"H_PPM","Y":"C_PPM"})
            hsqc_data = hsqc_data[["H_PPM","C_PPM","Intensity"]].copy().astype("float").round(hsqc_precision).sort_values(by=["H_PPM"], ascending=True)
            # hsqc_data["Identity"] = name
            hsqc_data.to_json(os.path.join(output_dir, f"{name}_HSQC_Preprocessed.json"),orient="records")
        if 'TOCSY' in data or 'tocsy' in data: 
            tocsy_data = pd.read_csv(os.path.join(Data_Path,data),skiprows=6)
            tocsy_data.X = tocsy_data.X.map(lambda x: x.rstrip('[ppm]'))
            tocsy_data.Y = tocsy_data.Y.map(lambda x: x.rstrip('[ppm]'))
            tocsy_data = tocsy_data.rename(columns={"X":"Ha","Y":"Hb"})
            tocsy_data = tocsy_data[['Ha','Hb','Intensity']].copy().astype("float").round(tocsy_precision).sort_values(by=["Ha"], ascending=True)
            # tocsy_data['Identity'] = name
            tocsy_data.to_json(os.path.join(output_dir, f"{name}_TOCSY_Preprocessed.json"),orient="records")
        else:
            pass
    return hsqc_data,tocsy_data
