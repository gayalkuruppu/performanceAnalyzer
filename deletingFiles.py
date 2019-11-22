import shutil

# use to delete specific set of files in different folders and subfolders

def path(folder, subfolder):
    return '/home/gayal/Documents/malith_project2/queueingTheoryTests/test_02_passthrough_and_echo/'+folder+'/'+subfolder+'/'


tests = ['echo-echo', 'echo-prime', 'prime-echo', 'prime-prime']
service = ['ballerina', 'netty']

for i in tests:
    for j in service:
        print(path(i, j)+'summary')
        shutil.rmtree(path(i, j)+'summary')
