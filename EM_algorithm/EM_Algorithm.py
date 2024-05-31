import numba as nb
import numpy as np
from numba import jit
from numba.core.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)

@jit
def E_step(train_image, Lambda, P, N, pixels):
    w=np.zeros((N,10))
    for i in range(N):
        total=0
        for j in range(10):
            w[i][j]=Lambda[j]
            for k in range(pixels):
                if(train_image[i][k]==1):
                    w[i][j]*=P[j][k]
                else:
                    w[i][j]*=(1-P[j][k])
            total+=w[i][j]
        if (total!=0):
            w[i, :]/=total
    
    return w

@jit
def M_step(train_image, w, N, pixels):
    Lambda_n=np.zeros((10))
    P_n=np.zeros((10,pixels))
    
    for i in range(10):
        temp=np.sum(w[:, i])
        Lambda_n[i]=temp/N
        for j in range(pixels):
            if(Lambda_n[i]!=0):
                P_n[i][j]=np.dot(train_image[:, j], np.transpose(w)[i, :])/temp
            else:
                P_n[i][j]=np.dot(train_image[:, j], np.transpose(w)[i, :])
    
    return Lambda_n, P_n

@jit
def prediction(train_image, train_label, Lambda, P, pixels):
    predict=np.zeros((10,10))
    predict_prob=np.zeros((10))
    for i in range(len(train_label)):
        for j in range(10):
            predict_prob[j]=Lambda[j]
            for k in range(pixels):
                if(train_image[i][k]==1):
                    predict_prob[j]*=P[j][k]
                else:
                    predict_prob[j]*=(1-P[j][k])
        predict[np.argmax(predict_prob)][train_label[i]]+=1

    return predict

@jit
def make_matching(predict):
    matching=np.zeros((10)).astype(np.int)
    matching2=np.zeros((10)).astype(np.int)
    for i in range(10):
        index=np.argmax(predict)
        group=index//10
        label=index%10
        #print(label, " ", group, "\n")
        matching[group]=label
        matching2[label]=group
        predict[:, label]=-1
        predict[group, :]=-1
        #print(predict)

    return matching,matching2 


@jit
def reprediction(train_image, train_label, Lambda, P, pixels, matching):
    predict=np.zeros((10,10))
    predict_prob=np.zeros((10))
    for i in range(len(train_label)):
        for j in range(10):
            predict_prob[j]=Lambda[j]
            for k in range(pixels):
                if(train_image[i][k]==1):
                    predict_prob[j]*=P[j][k]
                else:
                    predict_prob[j]*=(1-P[j][k])
        predict[matching[np.argmax(predict_prob)]][train_label[i]]+=1
    
    return predict

def show_image(P, image_row, image_col):
    for i in range(10):
        print("class{", i, "}:\n")
        for j in range(image_row):
            for k in range(image_col):
                if(P[i][j*28+k]>=0.5):
                    print(1, end=' ')
                else:
                    print(0, end=' ')
            print(" ")
        print("\n")

def show_final_image(P, matching, image_row, image_col):
    for i in range(10):
        print("label class {", i,"}:\n" )
        for j in range(image_row):
            for k in range(image_col):
                if(P[matching[i]][j*28+k]>=0.5):
                    print(1, end=" ")
                else:
                    print(0, end=" ")
            print(" ")
        print("\n")

def confusion(final_predict):
    confusion_matrix=np.zeros((10,4))
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if(i==j and i==k):
                    confusion_matrix[i][0]+=final_predict[j][k]
                elif(i==j):
                    confusion_matrix[i][1]+=final_predict[j][k]
                elif(i==k):
                    confusion_matrix[i][2]+=final_predict[j][k]
                else:
                    confusion_matrix[i][3]+=final_predict[j][k]
        
        print("Confusion Matrix ", i, ":\n")
        print("\t\t Predict number ", i, "Predict not number ", i)
        print("Is number ", i, "\t\t", confusion_matrix[i][0], "\t\t", confusion_matrix[i][1])
        print("Isn't number ", i, "\t", confusion_matrix[i][2], "\t", confusion_matrix[i][3])
        print("\nSensitivity (Successfully predict number ", i, ")\t :", confusion_matrix[i][0]/(confusion_matrix[i][0]+confusion_matrix[i][1]))
        print("\nSpecificity (Successfully predict not number ", i, ")\t :", confusion_matrix[i][3]/(confusion_matrix[i][2]+confusion_matrix[i][3]))
        print("\n")
        print('---------------------------------------------------------------\n')

if __name__ == "__main__":
    print("Loading...\n")
    train_image_file=open("train-images.idx3-ubyte","rb")
    train_label_file=open("train-labels.idx1-ubyte","rb")

    train_image_file.read(4)
    train_label_file.read(8)

    train_image_size=int.from_bytes(train_image_file.read(4), byteorder='big')
    image_row=int.from_bytes(train_image_file.read(4), byteorder='big')
    image_col=int.from_bytes(train_image_file.read(4), byteorder='big')

    train_image=np.zeros((train_image_size,image_row*image_col),dtype=int)
    train_label=np.zeros((train_image_size),dtype=int)

    for i in range(train_image_size):
        train_label[i]=int.from_bytes(train_label_file.read(1),byteorder='big')
        for j in range(image_row*image_col):
            train_image[i][j]=int.from_bytes(train_image_file.read(1),byteorder='big')/128

    train_image_file.close()
    train_label_file.close()

    print("Done\n")

    Lambda=np.full((10),0.1)
    Lambda_n=np.zeros((10))
    P=np.random.rand(10, image_row*image_col)
    P_n=np.zeros((10,image_row*image_col))
    delta_Lamdba=1
    delta_P=1

    count=1
    while(count<=20 and (delta_Lamdba>1e-5 or delta_P>1e-5)):
        w=E_step(train_image, Lambda, P, train_image_size, image_row*image_col)
        Lambda_n, P_n=M_step(train_image, w, train_image_size, image_row*image_col)
        show_image(P_n, image_row, image_col)
        delta_Lamdba=np.linalg.norm(Lambda-Lambda_n)
        delta_P=np.linalg.norm(P-P_n)
        print("No. of Iteration: ", count, ", Difference: ", delta_Lamdba+delta_P)
        print("-----------------------------------------\n")
        count+=1
        Lambda=Lambda_n
        P=P_n

    predict=prediction(train_image, train_label, Lambda, P, image_row*image_col)
    matching, matching2=make_matching(predict)
    final_predict=reprediction(train_image, train_label, Lambda, P, image_row*image_col,matching)
    show_final_image(P, matching2, image_row, image_col)
    confusion(final_predict)

    print("Total iteration to converge: ",count-1)
    print("Total error rate: ", delta_Lamdba+delta_P)
