import cv2
from PIL import ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def grabIM():
    return ImageGrab.grab()

def get_truth(image1,image2):
    '''if the two image's shapes are equivalent, return truth_matrix or else throw an exception
    The truth matrix is boolean matrix of either true or false if pixel intensities are the same'''
    gray_im1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray_im2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)   
    #gray_im1 = cv2.resize(gray_im1,(int(gray_im1.shape[1]/2),int(gray_im1.shape[0]/2,)))
    #gray_im2 = cv2.resize(gray_im2,(int(gray_im2.shape[1]/2),int(gray_im2.shape[0]/2)))
    
    print(gray_im1.shape,gray_im2.shape)
    if gray_im1.shape == gray_im2.shape:
         TAR_12 = gray_im1==gray_im2
         return TAR_12
            
    else:
         raise Exception('image shapes are not equivalent')
    

def evaluate_truth(truth_ar):
    '''get number of truths and falses and return indices of the false pixels'''
    tcount = 0 #the truth counts
    fcount = 0 #the false counts in the truth matrix
    idy = []
    idx = []
    diffs_idx =[] #get false counts indices
    for y in range(truth_ar.shape[0]):
        for x in range(truth_ar.shape[1]):
            if truth_ar[y][x] == True:
                tcount = tcount+1
            else:
                fcount = fcount+1
                diffs_idx.append([y,x])
                # get indices
                idx.append(x)
                idy.append(y)
    similarity = tcount/(tcount+fcount)
    print('ims are similar by',similarity)
    return (tcount,fcount,diffs_idx,idx,idy,similarity)



def getdiffs(false_mat,com_mat,idx,idy):
    # max and min false mat indices
    min_idx = np.array(idx).min()
    max_idx = np.array(idx).max()
    min_idy = np.array(idy).min()
    max_idy = np.array(idy).max()
    #rod = ((min_idx,min_idy),max_idx,max_idy)
    cv2.rectangle(com_mat[0],(min_idx,min_idy),(max_idx,max_idy),(0,0,255),2)
    cv2.imshow('frame',com_mat[0])
    cv2.waitKey(0)
    
    cv2.rectangle(com_mat[1],(min_idx,min_idy),(max_idx,max_idy),(0,0,255),2)
    cv2.imshow('frame',com_mat[1])
    cv2.waitKey(0)
    
def testrun():
    import time
    #im1 = cv2.imread('s1.png')
    #im2 = cv2.imread('s3.png')
    #im1 = cv2.resize(im1,(int(im1.shape[1]/2),int(im1.shape[0]/2,)))
    #im2 = cv2.resize(im2,(int(im2.shape[1]/2),int(im2.shape[0]/2)))

    try:
        im1 = np.array(grabIM())
        print('im1 grabbed')
        time.sleep(10) # time delay between screenshot grabs
        im2 = np.array(grabIM())
        print('im2 grabbed')
        truth = get_truth(im1,im2)
        print(truth)
        t12 = evaluate_truth(truth)
    except ValueError:
        raise Exception('Similarity Maxed')
    diffs_idx = t12[2]
    idx = t12[3]
    idy = t12[4]
    getdiffs(diffs_idx,[im1,im2],idx,idy)

if __name__ == "__main__":
    testrun()
    #print('im12',t12,'\n','im13',t13,'\n','im23',t23)