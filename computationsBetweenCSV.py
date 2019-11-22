import pandas as pd

path = '/home/gayal/Documents/malith_project2/test_results/balerinaTests/26_oct_2019/'
# tests = ['echo-echo', 'echo-prime', 'prime-echo', 'prime-prime']
# services = ['ballerina', 'netty']

tests = ['echo', 'prime']
services = ['ballerina']

concurrency = [1, 10, 50, 100, 500]
no_conc = len(concurrency)
test_mins = 8   # filtered no of test minutes


def compute(basepath, folder, subfolder):
    means = []
    df = pd.read_csv(basepath+folder+'/'+subfolder+'/'+'summary/summary.csv')
    for i in range(no_conc):
        average_latency = df['Mean'].iloc[i*test_mins:(i+1)*test_mins].mean(axis=0)
        means.append(average_latency)
    return means


def shortening_csv(basepath, tofolder):
    df = pd.read_csv(basepath+tofolder+'/summary.csv')
    newdf = df.filter(['Scenario Name', 'Concurrent Users', 'Message Size (Bytes)', 'Throughput (Requests/sec)',
                       'Average Response Time (ms)', 'Average Users in the System'], axis=1)
    newdf.to_csv(basepath+tofolder+'/summary_new.csv')


def add_to_csv(data, basepath, tofolder, service):
    df = pd.read_csv(basepath+tofolder+'/summary_new.csv')
    df[service] = data
    df.to_csv(basepath+tofolder+'/summary_new.csv')


for i in tests:
    shortening_csv(path, i)


for i in tests:
    for j in services:
        data = compute(path, i, j)
        add_to_csv(data, path, i, j)
