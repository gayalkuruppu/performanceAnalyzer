import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 22})


def mape(y_true, y_pred):
    # mean absolute percentage error
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true, y_pred):
    # root mean square error
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def lmbda(x):  # used to get lambda(m-1)
    total = 0
    for t in range(no_of_services):
        total += p_ratios[t] * ET[t][x]
    return (x + 1) / total


path = '/home/gayal/Documents/malith_project2/test_results/balerinaTests/25_oct_2019/withPrimeParsing/'
tests = ['echo-echo', 'echo-prime', 'prime-echo', 'prime-prime']
services = ['ballerina', 'netty']
conc = [1, 10, 50, 100, 500]
data = pd.read_csv(path+'bal100summary.csv')

test = 2

r_ee_test = data.iloc[5*(test-1):5*test, 4]
x_ee_test = data.iloc[5*(test-1):5*test, 3]
rm_ee_test = data.iloc[5*(test-1):5*test, 11]
rb_ee_test = data.iloc[5*(test-1):5*test, 7]

throughput = []
concurrency = 500


if test == 1:
    serviceRates = [50000, 8900, 12500]
    testname = 'Simple pass-through - Simple echo'
elif test == 2:
    serviceRates = [50000, 8900, 4400]
    testname = 'Simple pass-through - Prime echo'
elif test == 3:
    serviceRates = [50000, 4000, 12500]
    testname = 'Prime pass-through - Simple echo'
elif test == 4:
    serviceRates = [50000, 3800, 4450]
    testname = 'Prime pass-through - Prime echo'
else:
    print('invalid test case')

testname = 'Test case {} : {}'.format(test, testname)


no_of_services = len(serviceRates)  # number of servers
serverNames = ['JMeter Client', 'Pass-through Service', 'Backend Service', 'Total']
overheads = [0, 0, 0]   # overheads in seconds
# routingProb[i][j] = routing probability from i th server to j th server
routingProb = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]     # how to convert the routing Prob matrix into p_ratios
p_ratios = [0.25, 0.5, 0.25]
# p_ratios = [1/3, 1/3, 1/3]
# p_ratios = [1/2, 1/2]

p = 1/no_of_services
ET = np.zeros((no_of_services, concurrency))     # concurrency * no_of_servers 2d list
lm = np.arange(1, concurrency+1)        # nodes axis 1d list elements are concurrency from 1 to concurrency
EN = np.zeros((no_of_services, concurrency))     # concurrency * no_of_servers 2d list

for i in range(no_of_services):
    serverNames[i] = serverNames[i]+' (Service Rate = {} requests/sec) - MVA'.format(serviceRates[i])

for i in range(no_of_services):
    ET[i][0] = 1000*((1/serviceRates[i]) + overheads[i])    # expected time in milli seconds

# plt.figure(figsize=(16, 10))
# # plt.figure()
# plt.title('Response Time Vs Concurrency (analytical vs test) \n{}'.format(testname))
# plt.xlabel('Concurrency(N)')
# plt.ylabel('Response Time (ms)')
#

for n in range(1, concurrency):
    for s in range(no_of_services):
        ET[s][n] = (1 + p_ratios[s]*lmbda(n-1)*ET[s][n-1])*ET[s][0]

for n in range(1, concurrency):
    for s in range(no_of_services):
        EN[s][n] = ET[s][n]*lmbda(n)*p_ratios[s]

# plt.plot(conc, r_ee, label='Total Response Time - Test')
# plt.plot(conc, rm_ee, label='Pass-through Service - Test')
# plt.plot(conc, rb_ee, label='Backend Service - Test')

ER = np.sum(ET, axis=0)  # Response Time in milli seconds
'''
  E[s][n]
E[R]
for n in range(1, concurrency):
    for s in range(no_of_services):
        ET[s][n] = (1 + p_ratios[s]*lmbda(n-1)*ET[s][n-1])*ET[s][0]


'''

rm_ee_pred = []
rb_ee_pred = []

X = 1000*lm/ER  # Throughput req/sec
throughput.append(X)

for s in range(1, no_of_services):
    plt.plot(lm, ET[s], label=serverNames[s])

for c in conc:
    rm_ee_pred.append(ET[1][c-1])
    rb_ee_pred.append(ET[2][c-1])


print('-----------------------------------------------------')
print(testname)
print('-----------------------------------------------------')
print('MAPE for middle server = {}%'.format(mape(rm_ee_test, rm_ee_pred)))
print('RMSE for middle server = {}'.format(rmse(rm_ee_test, rm_ee_pred)))
print('-----------------------------------------------------')
print('MAPE for backend server = {}%'.format(mape(rb_ee_test, rb_ee_pred)))
print('RMSE for backend server = {}'.format(rmse(rb_ee_test, rb_ee_pred)))
print('-----------------------------------------------------')
