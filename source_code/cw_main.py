import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import pandas as pd

# Y holds classifications for training data
# X holds training data besides the classification
# P holds the costs which is from the testing data
Y = []
X = []
P = []

def read_and_predict():

    #Open the training data test file and assign it to the correct arrays
    with open('TrainingData.txt', 'r', encoding='UTF-8') as f:
        while (line := f.readline().rstrip()):
            t = line.split(",")
            t = np.asarray(t, dtype=float)
            Y.append(t[24])
            t=t[:24]
            X.append(t)
    f.close()
    
    #Train the classification model
    clf = svm.SVC()
    clf.fit(X, Y)

    #Open test data text file and assign it to P
    with open('TestingData.txt', 'r', encoding='UTF-8') as f:
        while (line := f.readline().rstrip()):
            t = line.split(",")
            t = np.asarray(t, dtype=float)
            P.append(t)
            
    f.close()
    
    #Using the model predict the classification of test data
    predict = clf.predict(P)
    
    #Using the model predict the classification of training data without it's classification data
    testerror = clf.predict(X)
    
    error = 0
    
    #Find percentage of training accuracy by counting the number of correct classifications comparing it to the classifications given before
    for i, test in enumerate(testerror):
        if test == Y[i]:
            error+=1
    perc = 100 * error/i
    
    #Write training accuracy value to a text file
    with open('error_percentage.txt', 'w') as f:
        f.truncate(0)
        f.write('Training Accuracy is {}'.format(perc) )
    f.close()
    
    #Write test results to a test file
    with open('TestingResults.txt', 'w') as f:
        f.truncate(0)
        for i, t in enumerate(P):
            result = ''
            for j in range(len(t)):
                result += str(t[j])
                result += ','
            result += str(int(predict[i]))
            f.write(result)
            f.write('\n')
    f.close()
    
    #return predicted values of test data
    return predict

def graphs_lp(predict):

    #Assign all user tasks from excel sheet to arrays
    excelFile = pd.read_excel ('COMP3217CW2Input.xlsx', sheet_name = 'User & Task ID')
    taskNames = excelFile['User & Task ID'].tolist()
    RT = excelFile['Ready Time'].tolist()
    DL = excelFile['Deadline'].tolist()
    EPH = excelFile['Maximum scheduled energy per hour'].tolist()
    ED = excelFile['Energy Demand'].tolist()
    tasks = []
    
    #Append the array tasks with the specific tasks and its parameters
    for k in range (len(RT)):
        task = []
        task.append(RT[k])
        task.append(DL[k])
        task.append(EPH[k])
        task.append(ED[k])
        tasks.append(task)
        
    for i, prices in enumerate(P):
        #Only abnormal classifications
        if (predict[i] == 1):

            V = []
            C = []
            E = []
            
            #LP Model for minimization
            model = LpProblem(name="Schedule", sense=LpMinimize)
            
            #Three for loops to write equation through going through each task for each user
            for j, task in enumerate(tasks):
                temp_list = []
                for k in range(task[0], task[1] + 1):
                    #Assign each LP variable a name which is a task in between the ready and deadline time
                    var = LpVariable(name=taskNames[j]+'_'+str(k), lowBound=0, upBound=task[2])
                    temp_list.append(var)
                V.append(temp_list)
            #Models the prices every hour using the maximum energy and price of the hour used that can be consumed  in LP
            for j, task in enumerate(tasks):
                for var in V[j]:
                    price = prices[int(var.name.split('_')[2])]
                    C.append(price * var)
            
            model += lpSum(C)
            
            #Models the amount of energy each task uses in LP
            for j, task in enumerate(tasks):
                temp_list = []
                for var in V[j]:
                    temp_list.append(var)
                E.append(temp_list)
                model += lpSum(temp_list) == task[3]
            
            #Solve the cost LP model
            model.solve()
            
            #Plots bar charts of each day consisting of hours against energy used in total with user breakdown consumptions
            hours = [str(x) for x in range(0, 24)]
            pos = np.arange(len(hours))
            users = ['user1', 'user2', 'user3', 'user4', 'user5']
            color_list = ['red','orange','yellow','green','blue']
            plot_list = []

            for user in users:
                temp_list = []
                for hour in hours:
                    hour_list_temp = []
                    task_count = 0
                    for var in model.variables():
                        if user == var.name.split('_')[0] and str(hour) == var.name.split('_')[2]:
                            task_count += 1
                            hour_list_temp.append(var.value())
                    temp_list.append(sum(hour_list_temp))
                plot_list.append(temp_list)

            plt.bar(pos,plot_list[0],color=color_list[0],bottom=0)
            plt.bar(pos,plot_list[1],color=color_list[1],bottom=np.array(plot_list[0]))
            plt.bar(pos,plot_list[2],color=color_list[2],bottom=np.array(plot_list[0])+np.array(plot_list[1]))
            plt.bar(pos,plot_list[3],color=color_list[3],bottom=np.array(plot_list[0])+np.array(plot_list[1])+np.array(plot_list[2]))
            plt.bar(pos,plot_list[4],color=color_list[4],bottom=np.array(plot_list[0])+np.array(plot_list[1])+np.array(plot_list[2])+np.array(plot_list[3]))
    
            plt.xticks(pos, hours)
            plt.xlabel('Hour')
            plt.ylabel('Energy Usage (kW)')
            plt.title('Energy Usage Per Hour For All Users\nDay {}'.format(i+1))
            plt.legend(users,loc=0)
            #Saves figure to a folder
            plt.savefig('graph/{}'.format(i+1))
            plt.clf()

#Execution of both functions
predict = read_and_predict()
graphs_lp(predict)