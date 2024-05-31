import numpy as np
import matplotlib.pyplot as plt

# compute A=LU and return U^(-1)*L^(-1)=A^(-1)
def inverse(A,n):
    
    #initial
    L=np.zeros((n,n))
    U=np.zeros((n,n))
    INV_L=np.zeros((n,n))
    INV_U=np.zeros((n,n))

    #compute 1st row of U and 1st column and diagonal entries of L
    for i in range(n):
        U[0][i]=A[0][i]
        L[i][0]=A[i][0]/U[0][0]
        L[i][i]=1

    #compute other entries
    for i in range(1,n):
        for j in range(1,n):
            temp=0
            for k in range(0,i):
                temp=temp+L[i][k]*U[k][j]
            U[i][j]=A[i][j]-temp
        for j in range(i+1,n):
            temp=0
            for k in range(0,i):
                temp=temp+L[j][k]*U[k][i]
            L[j][i]=(A[j][i]-temp)/U[i][i]

    #compute inverse of U
    for i in range(n):
        INV_U[i][i]=1/U[i][i]
        for j in range(i-1,-1,-1):
            temp=0
            for k in range(j+1,i+1):
                temp=temp+U[j][k]*INV_U[k][i]
            INV_U[j][i]=-temp/U[j][j]

    #compute inverse of L
    for i in range(n):
        INV_L[i][i]=1
        for j in range(i+1,n):
            INV_L[j][i]=-L[j][i]

    #return U^(-1)*L^(-1)
    return INV_U@INV_L

# compute vector x that produce less errors by using Newton method
def newton_method(A,n,b):
    u0=np.ones((n,1))
    diff=1
    iteration=0
    while diff>1e-5 and iteration<100:
        u1=u0-inverse(2*A_T@A,n)@(2*A_T@A@u0-2*A_T@b)
        diff=abs(np.sum(np.square(u1-u0))/n)
        iteration+=1
        u0=u1
    return u0

# compute vector x that produce less errors by using gradient descent method
def steep_method(A,n,b,l):
    learning_rate=1e-4
    u0=np.ones((n,1))
    diff=1
    while diff>1e-7:
        u1=u0-learning_rate*(2*A_T@A@u0-2*A_T@b+l*np.ones((n,1)))
        diff=abs(np.sum(np.square(u1-u0))/n)
        u0=u1
    return u0

# compute the loss for each method
def loss(A,x,b):
    return np.sum(np.square(A@x-b))

# plot the regression and the points of primal data
def visualization(matrix,lse_x,newton_x,steep_x,left,right,n):
    x=np.arange(left,right,0.1)
    lse_y=0
    steep_y=0
    newton_y=0

    for i in range (n-1,-1,-1):
        lse_y=lse_y+lse_x[i]*np.power(x,i)
        steep_y=steep_y+steep_x[i]*np.power(x,i)
        newton_y=newton_y+newton_x[i]*np.power(x,i)

    
    plt.subplot(3,1,1)
    plt.plot(matrix[:,0],matrix[:,1],'ro',markeredgecolor='k')
    plt.plot(x,lse_y,'-k')
    plt.subplot(3,1,2)
    plt.plot(matrix[:,0],matrix[:,1],'ro',markeredgecolor='k')
    plt.plot(x,steep_y,'-k')
    plt.subplot(3,1,3)
    plt.plot(matrix[:,0],matrix[:,1],'ro',markeredgecolor='k')
    plt.plot(x,newton_y,'-k')
    plt.show()

# main function
if __name__ == '__main__':
    # read and get the data
    with open('testfile.txt', 'r') as file:
        file_content = file.read()
    
    data = [float(num) for num in file_content.replace(',', ' ').split()]
    num_data=int(len(data)/2)
    matrix = np.array(data).reshape((num_data, 2))

    # enter inputs n and lamdba
    n=int(input("n: "))
    l=float(input("lambda: "))
    
    # produce the matrix A 
    A=np.empty((num_data,n))
    for i in range(n):
        A[:,i]=np.power(matrix[:,0],i).reshape(-1)
    
    A_T=np.transpose(A)
    b=matrix[:,1].reshape((num_data,1))
  
    # compute each x and error by using different method
    lse_x=inverse(A_T@A+l*np.eye(n),n)@A_T@b
    loss_lse=loss(A,lse_x,b)

    steep_x=steep_method(A,n,b,l)
    loss_steep=loss(A,steep_x,b)

    newton_x=newton_method(A,n,b)
    loss_newton=loss(A,newton_x,b)

    # show the polynomial and the total error 
    print("LSE: \nFitting line: ")
    for i in range(n-1,0,-1):
        print(lse_x[i],'X^',i,' + ',end='')
    print(lse_x[0])
    print("Total error: ",loss_lse,"\n")

    print("steepest descent Method:\nFitting line: ")
    for i in range(n-1,0,-1):
        print(steep_x[i],'X^',i,' + ',end='')
    print(steep_x[0])
    print("Total error: ",loss_steep,"\n")

    print("Newton's Method:\nFitting line: ")
    for i in range(n-1,0,-1):
        print(newton_x[i],'X^',i,' + ',end='')
    print(newton_x[0])
    print("Total error: ",loss_newton,"\n")

    left=np.min(matrix[:,0])-1.0
    right=np.max(matrix[:,0])+1.0
    
    # show the figure
    visualization(matrix,lse_x,newton_x,steep_x,left,right,n)



    