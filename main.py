from mysvd import csvd,rsvd,normal_SVD
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import time

def func(img,i,k,p,n_iter):

    A=img[:,:,i]
    # print(A.dtype)
    A_rsvd=0
    A_csvd=0
    A_nsvd=0
    time_calc=[]
    err=[]
    start_time = time.time()
    A_rsvd,U,S,V_t =rsvd(A,k,p,n_iter)
    time_calc.append((time.time() - start_time))
    err.append(la.norm(A - A_rsvd, 2))

    start_time2= time.time()
    A_csvd,U, S, V= csvd(A,k,p,n_iter)
    time_calc.append((time.time() - start_time2))
    err.append(la.norm(A - A_csvd, 2))

    start_time3= time.time()
    A_nsvd,U, S, V=normal_SVD(A,k,p,n_iter)
    err.append(la.norm(A - A_nsvd, 2))
    time_calc.append((time.time() - start_time3))
    return A_rsvd,A_csvd,A_nsvd,err,time_calc

def func_gray(img,k,p,n_iter):

    A=img
    # print(A.dtype)
    A_rsvd=0
    A_csvd=0
    time_calc=[]
    err=[]
    start_time = time.time()
    A_rsvd,U,S,V_t =rsvd(A,k,p,n_iter)
    time_calc.append((time.time() - start_time))
    err.append(la.norm(A - A_rsvd, 2))

    start_time2= time.time()
    A_csvd,U, S, V= csvd(A,k,p,n_iter)
    time_calc.append((time.time() - start_time2))
    err.append(la.norm(A - A_csvd, 2))

    start_time3= time.time()
    A_nsvd,U, S, V=normal_SVD(A,k,p,n_iter)
    err.append(la.norm(A - A_nsvd, 2))
    time_calc.append((time.time() - start_time3))
    return A_rsvd,A_csvd,A_nsvd,err,time_calc


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def myfunc_gray(img,k,p,n_iter):
    rsvd,csvd,nsvd,err,calc_time=func_gray(img,k,p,n_iter)
    rsvd=np.array(rsvd,dtype=np.uint8)
    csvd=np.array(csvd,dtype=np.uint8)
    nsvd=np.array(nsvd,dtype=np.uint8)
    # plt.imshow((rsvd))
    # plt.imsave("images/RSVDimg"+str(k)+".jpg",rsvd)
    # plt.show()
    # plt.imshow((csvd))
    # plt.imsave("images/CSVDimg"+str(k)+".jpg",csvd)
    # plt.show()
    # plt.imshow((nsvd))
    # plt.imsave("images/NSVDimg"+str(k)+".jpg",nsvd)
    # plt.show()
    # print(err)
    # print(calc_time)
    return err,calc_time

def getplotforgray(img,k,p,n_iter):

    img= rgb2gray(img)
    K=[]
    print(img.shape)
    for k in range(1000,4001,400):
        print(k)
        K.append(k)
        err,calc_time=myfunc_gray(img,k,p,n_iter)
        rsvd_err.append(err[0])
        csvd_err.append(err[1])
        nsvd_err.append(err[2])
        rsvd_time.append(calc_time[0])
        csvd_time.append(calc_time[1])
        nsvd_time.append(calc_time[2])

    print(rsvd_time,rsvd_err)
    print(csvd_time,csvd_err)
    print(nsvd_time,nsvd_err)

    # K = [10,100,1000,10000]

    plt.plot(K, rsvd_time, color='r',marker="*")
    plt.plot(K, csvd_time, color='g',marker=".")
    plt.plot(K, nsvd_time, color='b',marker="+")
    plt.xlabel('Target rank')
    plt.ylabel('Time taken')
    plt.legend(('RSVD', 'CSVD','NSVD'))
    # plt.ylim((.70, 1))
    plt.show()




    plt.plot(K, rsvd_err, color='r',marker="*")
    plt.plot(K, csvd_err, color='g',marker=".")
    plt.plot(K, nsvd_err, color='b',marker="+")
    plt.xlabel('Target rank')
    plt.ylabel('Error')
    plt.legend(('RSVD', 'CSVD','NSVD'))
    # plt.ylim((.70, 1))
    plt.show()
def getreconstruction(img,k,p,n_iter,file):
 
    Rrsvd,Rcsvd,Rnsvd,err,calc_time=func(img,0,k,p,n_iter)
    Rrsvd=np.array(Rrsvd,dtype=np.uint8)
    Rcsvd=np.array(Rcsvd,dtype=np.uint8)
    Rnsvd=np.array(Rnsvd,dtype=np.uint8)

    Grsvd,Gcsvd,Gnsvd,err,calc_time=func(img,1,k,p,n_iter)
    Grsvd=np.array(Grsvd,dtype=np.uint8)
    Gcsvd=np.array(Gcsvd,dtype=np.uint8)
    Gnsvd=np.array(Gnsvd,dtype=np.uint8)


    Brsvd,Bcsvd,Bnsvd,err,calc_time=func(img,2,k,p,n_iter)
    Brsvd=np.array(Brsvd,dtype=np.uint8)
    Bcsvd=np.array(Bcsvd,dtype=np.uint8)
    Bnsvd=np.array(Bnsvd,dtype=np.uint8)


    RSVDimg=(np.dstack((Rrsvd,Grsvd,Brsvd)))
    CSVDimg=(np.dstack((Rcsvd,Gcsvd,Bcsvd)))
    NSVDimg=(np.dstack((Rnsvd,Gnsvd,Bnsvd)))

    plt.imshow((RSVDimg))
    plt.imsave("images/RSVDimg_"+file+".jpg",RSVDimg)
    # plt.show()
    plt.imshow((CSVDimg))
    plt.imsave("images/CSVDimg_"+file+".jpg",CSVDimg)
    # plt.show()
    plt.imshow((NSVDimg))
    plt.imsave("images/NSVDimg_"+file+".jpg",NSVDimg)
    # plt.show()

    print(err)
    print(calc_time)

if __name__ == "__main__":

    rsvd_err=[]
    csvd_err=[]
    nsvd_err=[]

    rsvd_time=[]
    csvd_time=[]
    nsvd_time=[]
    k = 1500
    p = 10
    n_iter=0
    file="fullsize.jpg"
    img=image.imread(file)
    #to get plot for gray
    # getplotforgray(img,k,p,n_iter)
    getreconstruction(img,k,p,n_iter,file)

    