import numpy as np 
import math

if __name__ == '__main__':
    lines=[]
    with open('testfile.txt', 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())

    a=int(input("a: "))
    b=int(input("b: "))

    for i in range(len(lines)):
        ones=0
        for j in range(len(lines[i])):
            if(lines[i][j]=='1'):
                ones+=1
        prob=ones/len(lines[i])
        likelihood=math.factorial(len(lines[i]))/math.factorial(ones)/math.factorial(len(lines[i])-ones)*(prob**ones)*((1-prob)**(len(lines[i])-ones))
        print("case: ",i+1,": ",lines[i],"\n")
        print("Likelihood: ",likelihood)
        print("Beta prior:     a = ",a," b = ",b)
        a+=ones
        b+=(len(lines[i])-ones)
        print("Beta posterior: a = ",a," b = ",b,"\n")