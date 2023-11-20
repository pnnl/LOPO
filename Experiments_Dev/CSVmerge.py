import torch
import numpy as np
import pandas as pd
import pickle
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dir",    type=str,    default='none',  help="CSV directory path" )
args = parser.parse_args()

dframes = []
path =  './'+args.dir+'/'
with os.scandir( path ) as dir:
    for entry in dir:
        if entry.name.endswith(".csv") and entry.is_file():

            print( entry.name )
            dframes.append( pd.read_csv( path+entry.name ) )


dframes_cat = pd.concat(dframes)


print("dframes_cat = ")
print( dframes_cat )

dframes_cat.to_csv(args.dir+'.csv')
#dframes_cat.to_csv('js_csv1019cp.csv')
