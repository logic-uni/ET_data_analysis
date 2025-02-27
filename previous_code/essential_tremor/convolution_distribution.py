import csv
import matplotlib.pyplot as plt
import numpy as np

a=[]
b=[]
c=[]
with open('C:/Users/zyh20/Desktop/csv/-0.1to0ET.csv', 'r') as file:
    csv_reader = csv.reader(file)
    
    for row in csv_reader:
        a.append(float(row[1]))

with open('C:/Users/zyh20/Desktop/csv/0to5ET.csv', 'r') as file:
    csv_reader = csv.reader(file)
    
    for row in csv_reader:
        b.append(float(row[1]))

with open('C:/Users/zyh20/Desktop/csv/-0.1to5ET.csv', 'r') as file:
    csv_reader = csv.reader(file)
    
    for row in csv_reader:
        c.append(float(row[1]))

for i in range((len(b)-len(a))):
    a.append(0)


a=a[::-1]


x1=[i for i in range(0, len(a), 1)]
plt.bar(x1, a)
plt.show()

x2=[i for i in range(0, len(b), 1)]
plt.bar(x2, b)
plt.show()

x3=[i for i in range(0, len(c), 1)]
plt.bar(x3, c)
plt.show()

d=np.convolve(a, c, mode='full')
x4=[i for i in range(0, len(d), 1)]
plt.bar(x4, d)
plt.show()
