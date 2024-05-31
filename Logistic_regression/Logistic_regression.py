import numpy as np
import matplotlib.pyplot as plt
import math

def univariate_gaussian(mean, var):
    uniform_value=np.random.uniform(0,1,size=12)
    result=np.sum(uniform_value)-6
    return result*np.sqrt(var)+mean

def gradient_descent_method(X, group):
    w=np.ones((3,1))
    delta_J=np.ones((3,1))
    count = 0
    while(np.linalg.norm(delta_J)>=0.01 and count<1e4):
        delta_J=(np.transpose(X)@(group-1/(1+np.exp(-X@w))))
        w+=delta_J
        count+=1
    return w

def newton_method(X, group):
    w=np.ones((3,1))
    D=np.zeros((len(group),len(group)))
    delta_J=np.ones((3,1))
    count=0

    while(np.linalg.norm(delta_J)>=0.01 and count<1e4):
        delta_J=(np.transpose(X)@(group-1/(1+np.exp(-X@w))))
        check=True
        for i in range(len(group)):
            temp=np.exp(-X[i]@w)
            if (temp>100):
                check=False
            D[i][i]=temp/(1+np.power(temp,2))
        
        Hf=np.transpose(X)@D@X

        if (check and np.linalg.det(Hf)!=0):
            delta_J=np.linalg.inv(Hf)@delta_J

        w+=delta_J
        count+=1

    return w

def printResult(w, X, group):
    predict=np.zeros((len(group), 1))
    confusion=np.zeros((2, 2))

    for i in range(len(group)):
        if(X[i]@w>=0):
            predict[i][0]=1
        confusion[int(group[i][0])][int(predict[i][0])]+=1

    print("w:\n")
    for i in range(len(w)):
        print(w[i][0],"\n")
    print("\n Confusion Matrix:\n")
    print("\t\t\t Is cluster 1\t Is cluster 2")
    print("Predict cluster 1\t", confusion[0][0], "\t\t", confusion[0][1])
    print("Predict cluster 2\t", confusion[1][0], "\t\t", confusion[1][1])
    print("\nSensitivity (Successfully predict cluster 1): ", confusion[0][0]/(confusion[0][0]+confusion[0][1]))
    print("Specificity (Successfully predict cluster 2): ", confusion[1][1]/(confusion[1][0]+confusion[1][1]))

    return predict

def visualization(X, predict, title, index):
    plt.subplot(1,3,index)
    plt.title(title)
    
    for i in range(len(predict)):
        if predict[i][0]==0:
            plt.scatter(X[i][0], X[i][1], color='red', s=10)
        else:
            plt.scatter(X[i][0], X[i][1], color='blue', s=10)
    

if __name__ == '__main__':
    N=int(input("N: "))
    mx1=float(input("mean of x in D1: "))
    vx1=float(input("variance of x in D1: "))
    my1=float(input("mean of y in D1: "))
    vy1=float(input("variance of y in D1: "))
    mx2=float(input("mean of x in D2: "))
    vx2=float(input("variance of x in D2: "))
    my2=float(input("mean of y in D2: "))
    vy2=float(input("variance of y in D2: "))

    X=np.ones((2*N,3))
    group=np.zeros((2*N,1))

    for i in range(N):
        X[i][0]=univariate_gaussian(mx1, vx1)
        X[i][1]=univariate_gaussian(my1, vy1)
    
    for i in range(N,2*N):
        X[i][0]=univariate_gaussian(mx2, vx2)
        X[i][1]=univariate_gaussian(my2, vy2)
        group[i]=1

    w_g=gradient_descent_method(X, group)
    w_n=newton_method(X, group)
    
    print("Gradient descent: \n\n")
    predict_g=printResult(w_g, X, group)
    print("-------------------------------------\n")
    print("Newton's method: \n\n")
    predict_n=printResult(w_n, X, group)
    
    visualization(X, group, "Ground truth", 1)
    visualization(X, predict_g, "Gradient descent", 2)
    visualization(X, predict_n, "Newton's method", 3)

    plt.tight_layout()
    plt.show()