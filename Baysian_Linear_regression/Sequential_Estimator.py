import numpy as np
import matplotlib.pyplot as plt
import math

def univariate_gaussian(mean, var):
    uniform_value=np.random.uniform(0,1,size=12)
    result=np.sum(uniform_value)-6
    return result*np.sqrt(var)+mean

if __name__ == '__main__':
    m=float(input("mean: "))
    s=float(input("variance: "))
    print("Data point source function: N(",m,", ",s,")\n")
    mean=univariate_gaussian(m,s)
    count=1
    M2=0
    print("Add data point: ",mean)
    print("Mean = ",mean,"      Variance = ",M2)

    while(1):
        add_point=univariate_gaussian(m,s)
        count+=1
        delta=add_point-mean
        mean+=delta/count
        M2+=(delta)*(add_point-mean)
        print("Add data point: ",add_point)
        print("Mean = ",mean,"      Variance = ",M2/(count-1))
        if(abs(mean-m)<1e-2 and abs((M2/(count-1))-s)<1e-2):
            break


