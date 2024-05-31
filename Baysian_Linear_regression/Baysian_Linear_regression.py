import numpy as np
import matplotlib.pyplot as plt
from Sequential_Estimator import univariate_gaussian
import math

def polynomial_generator(w,n,a):
    x=np.random.uniform(-1,1)
    y=univariate_gaussian(0,a)
    for i in range(n):
        y+=w[i]*np.power(x,i)
    return x, y

def visualization(point_x,point_y,title,pos,mean,cov,n):
    x=np.linspace(-2.0, 2.0, 100)
    func=np.poly1d(np.flip(np.reshape(mean, n)))
    y=func(x)
    phi=np.zeros((1,n))        
    plt.subplot(2,2,pos)
    plt.title(title)
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 25.0)
    plt.scatter(point_x,point_y)
    var=np.zeros((100))
    for i in range(100):
        for j in range(n):
            phi[0][j]=np.power(x[i],j)
        var[i]=a+phi@cov@np.transpose(phi)

    plt.plot(x,y,'k')
    plt.plot(x,y+var,'r')
    plt.plot(x,y-var,'r')


if __name__ == '__main__':
    b=float(input("b= "))
    n=int(input("n= "))
    a=float(input("a= "))
    w=np.array([int(input(f"")) for i in range(n)])
    
    x=np.linspace(-2.0, 2.0, 100)
    func=np.poly1d(np.flip(w))
    y=func(x)
    plt.subplot(2,2,1)
    plt.title("Ground truth")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 25.0)
    plt.plot(x,y,'k')
    plt.plot(x,y+a,'r')
    plt.plot(x,y-a,'r')
    
    count=0
    point_x=[]
    point_y=[]
    prior_covariance=np.zeros((n,n))
    prior_mean=np.zeros((n,1))

    while(True):
        add_x, add_y=polynomial_generator(w,n,a)
        #add_x=float(input("x= "))
        #add_y=float(input("y= "))
        print("Add data point (",add_x,", ",add_y,"):\n")
        point_x.append(add_x)
        point_y.append(add_y)
        phi=np.zeros((1,n))        
        for i in range(n):
            phi[0][i]=np.power(add_x,i)

        if(count):  
            covariance=a*np.transpose(phi)@phi+prior_covariance
            mean=np.linalg.inv(covariance)@(a*add_y*np.transpose(phi)+prior_covariance@prior_mean)
        else:
            covariance=a*np.transpose(phi)@phi+b*np.eye(n)
            mean=a*np.linalg.inv(covariance)@np.transpose(phi)*add_y

        predict_mean=phi@mean
        predict_variance=a+phi@np.linalg.inv(covariance)@np.transpose(phi)

        print("Posterior mean:\n")
        for i in range(n):
            print(mean[i],"\n")
        print("Posterior variance:\n")
        inverse_covariance=np.linalg.inv(covariance)
        for i in range(n):
            for j in range(n):
                print(inverse_covariance[i][j], end=" ")
            print(" ")
        print("Predictive distribution ~ N(",predict_mean,", ",predict_variance,")\n")

        count+=1
        
        if(count==10):
            visualization(point_x,point_y,"After 10 incomes",3,mean,np.linalg.inv(covariance),n)
        if(count==50):
            visualization(point_x,point_y,"After 50 incomes",4,mean,np.linalg.inv(covariance),n)

        if(np.linalg.norm(mean-prior_mean)<1e-6 and count>100):
            break
        
        prior_covariance=covariance
        prior_mean=mean
    
    visualization(point_x,point_y,"Predict result",2,mean,np.linalg.inv(covariance),n)
    plt.tight_layout()
    plt.show()

