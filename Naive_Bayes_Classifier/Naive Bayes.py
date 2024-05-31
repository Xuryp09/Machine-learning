import numba as nb
import numpy as np
import math

@nb.njit()
def Discrete(train_image,train_label,test_image, test_label, image_size):
    
    prior=np.zeros((10),dtype=np.float64)
    for i in range(len(train_label)):
        prior[train_label[i]]+=1
    prior/=len(train_label)
    
    likelihood=np.zeros((10,image_size,32),dtype=np.float64)   
    likelihood_total=np.zeros((10,image_size),dtype=np.float64)

    for i in range(len(train_label)):
        for j in range(image_size):
            likelihood[train_label[i]][j][math.floor(train_image[i][j]/8)]+=1
    
    for i in range(10):
        for j in range(image_size):
            for k in range(32):
                likelihood_total[i][j]+=likelihood[i][j][k]
            likelihood[i][j][:]/=likelihood_total[i][j]
    
    for i in range(10):
        for j in range(image_size):
            for k in range(32):
                if(likelihood[i][j][k]==0):
                    likelihood[i][j][k]=1e-4
    
    posterior=np.zeros((len(test_label),10),dtype=np.float64)
    prediction=np.zeros((len(test_label)),dtype=np.int64)
    error=0

    for i in range(len(test_label)):
        posterior[i]=np.log(prior)
        for j in range(10):
            for k in range(image_size):
                posterior[i][j]+=np.log(likelihood[j][k][math.floor(test_image[i][k]/8)])

        posterior[i]/=np.sum(posterior[i])
        prediction[i]=np.argmin(posterior[i])
        if(prediction[i]!=test_label[i]):
            error+=1

    error/=len(test_label)
    return posterior, prediction, likelihood, error

def show_discrete(test_label, posterior, prediction, likelihood, error):
    for i in range(len(test_label)):
        print("Posterior (in log scale):")
        for j in range(10):
            print(j,": ",posterior[i][j])
        print("Prediction: ",prediction[i],", Ans: ",test_label[i],"\n")
    
    print("Imagination of numbers in Bayesian classifier: \n")
    for i in range(10):
        print(i,":\n")
        for j in range(28):
            for k in range(28):
                color=0
                color+=sum(likelihood[i][j*28+k][:17])
                color-=sum(likelihood[i][j*28+k][17:])
                print(f'{0 if color>0 else 1} ', end='')
            print("\n")

    print("Error rate: ",error)

@nb.njit()
def Continuous(train_image,train_label,test_image,test_label,image_size):
    prior=np.zeros((10),dtype=np.float64)
    for i in range(len(train_label)):
        prior[train_label[i]]+=1

    mean=np.zeros((10,image_size),dtype=np.float64)
    variance=np.zeros((10,image_size),dtype=np.float64)

    for i in range(len(train_label)):
        for j in range(image_size):
            mean[train_label[i]][j]+=train_image[i][j]
            variance[train_label[i]][j]+=np.square(train_image[i][j])

    for i in range(10):
        mean[i][:]/=prior[i]
        variance[i][:]/=prior[i]
        variance[i][:]-=np.square(mean[i][:])
    
    prior/=len(train_label)
    posterior=np.zeros((len(test_label),10),dtype=np.float64)
    prediction=np.zeros((len(test_label)),dtype=np.int64)
    error=0

    for i in range(len(test_label)):
        posterior[i]=np.log(prior)
        for j in range(10):
            for k in range(image_size):
                if (variance[j][k]!=0):
                    posterior[i][j]-=(0.5*np.log(2*np.pi*variance[j][k])+np.square(test_image[i][k]-mean[j][k])/2.0/variance[j][k])
        
        posterior[i]/=np.sum(posterior[i])
        prediction[i]=np.argmin(posterior[i])
        if(prediction[i]!=test_label[i]):
            error+=1
            
    error/=len(test_label)
    return posterior, prediction, mean, error

def show_continuous(test_label, posterior, prediction, mean, error):
    for i in range(len(test_label)):
        print("Posterior (in log scale):")
        for j in range(10):
            print(j,": ",posterior[i][j])
        print("Prediction: ",prediction[i],", Ans: ",test_label[i],"\n")
    
    print("Imagination of numbers in Bayesian classifier: \n")
    for i in range(10):
        print(i,":\n")
        for j in range(28):
            for k in range(28):
                print(f'{1 if mean[i][j*28+k]>128 else 0} ', end='')
            print("\n") 
        
    print("Error rate: ",error)

if __name__ == "__main__":
    train_image_file=open("train-images.idx3-ubyte","rb")
    train_label_file=open("train-labels.idx1-ubyte","rb")
    test_image_file=open("t10k-images.idx3-ubyte","rb")
    test_label_file=open("t10k-labels.idx1-ubyte","rb")

    train_image_file.read(4)
    test_image_file.read(4)
    train_label_file.read(8)
    test_label_file.read(8)

    train_image_size=int.from_bytes(train_image_file.read(4), byteorder='big')
    test_image_size=int.from_bytes(test_image_file.read(4), byteorder='big')
    image_row=int.from_bytes(train_image_file.read(4), byteorder='big')
    image_col=int.from_bytes(train_image_file.read(4), byteorder='big')
    test_image_file.read(8)

    train_image=np.zeros((train_image_size,image_row*image_col),dtype=int)
    train_label=np.zeros((train_image_size),dtype=int)
    test_image=np.zeros((test_image_size,image_row*image_col),dtype=int)
    test_label=np.zeros((test_image_size),dtype=int)

    for i in range(train_image_size):
        train_label[i]=int.from_bytes(train_label_file.read(1),byteorder='big')
        for j in range(image_row*image_col):
            train_image[i][j]=int.from_bytes(train_image_file.read(1),byteorder='big')
        
    for i in range(test_image_size):
        test_label[i]=int.from_bytes(test_label_file.read(1),byteorder='big')
        for j in range(image_row*image_col):
            test_image[i][j]=int.from_bytes(test_image_file.read(1),byteorder='big')

    train_image_file.close()
    train_label_file.close()
    test_image_file.close()
    test_label_file.close()

    option=int(input("mode: "))
    if(option==0):
        posterior, prediction, likelihood, error = Discrete(train_image,train_label,test_image,test_label,image_row*image_col)
        show_discrete(test_label, posterior, prediction, likelihood, error)
    else:
        posterior, prediction, mean, error = Continuous(train_image,train_label,test_image,test_label,image_row*image_col)
        show_continuous(test_label, posterior, prediction, mean, error)
