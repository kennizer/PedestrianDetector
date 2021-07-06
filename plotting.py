import sys 
import matplotlib.pyplot as plt 

x_axis = []
for i in range(40):
    x_axis.append(i-1) 
problem1_file = sys.argv[1]
problem2_file = sys.argv[2]
loss_1 = [] 
accuracy_1 = [] 
loss_2 = [] 
accuracy_2 = [] 
f = open(problem1_file, "r")
for line in f: 
    line_list = line.split()
    if line_list[0]== "Test:":
        loss_1.append(float(line_list[4]))
        accuracy_1.append(float(line_list[6]))
f = open(problem2_file, "r")
for line in f: 
    line_list = line.split()
    if line_list[0]== "Test:":
        loss_2.append(float(line_list[4]))
        accuracy_2.append(float(line_list[6]))
plt.plot(x_axis, loss_2) 
plt.show() 