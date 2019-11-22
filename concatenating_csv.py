import pandas as pd
import glob
import os


def concat(path, folder, subfolder):
    path += folder+'/'+subfolder+'/'
    csvfiles = sorted(glob.glob(path+'*.csv'))

    df = pd.DataFrame()
    for files in csvfiles:
        df = df.append(pd.read_csv(files).iloc[1:9])

    os.mkdir(path+'summary')
    df.to_csv(path+'summary/summary.csv', index=False)


path = '/home/gayal/Documents/malith_project2/test_results/balerinaTests/26_oct_2019/'
# tests = ['echo-echo', 'echo-prime', 'prime-echo', 'prime-prime']
# service = ['ballerina', 'netty']
tests = ['echo', 'prime']
service = ['ballerina']


for i in tests:
    for j in service:
        concat(path, i, j)
