import pandas as pd

file = ('test_file')
file_extension = ('.xlsx')
columns_to_interpolate = [1,2,4] #Columns start at 0

#Can take multiple filetypes as indata but will only output xlsx(only xlsx has been tested as of now)
if file_extension == '.xlsx' or file_extension == '.xls' or file_extension == '.xlsm' or file_extension == '.xlsb' or file_extension == '.odf' or file_extension == '.ods' or file_extension == '.odt':
    Data = pd.read_excel(file+file_extension)
elif file_extension == '.csv':
    Data = pd.read_csv(file+file_extension)
elif file_extension == '.pkl':
    Data = pd.read_pickle(file+file_extension)
elif file_extension == '.json':
    Data = pd.read_json(file+file_extension)
else:
    raise Exception("Not a valid file extension")



#Interpolate selected columns
for i in range(0,max(columns_to_interpolate)+1):
    if i in columns_to_interpolate:
        Data.iloc[:,i].interpolate(method='linear', axis=0, inplace=True)

#Fill in any empty cells with 0 that we did not want to interpolate
Data.fillna(0)

#Only keep rows where all the values are greater than 0
Data = Data[(Data > 0).all(axis=1)]

Data.to_excel(file+'_filtered.xlsx', index=False)
