import pandas as pd
import os 

def load_table(table):
   '''  
   download automaticly our data base
   '''
   #go up a notch to access the "data" folder
   os.chdir("../")
   os.chdir('data')
   csv_file_name = f'{table}.csv'
   table=pd.read_csv(csv_file_name)
   return table