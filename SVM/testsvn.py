#Reference https://gist.github.com/mblondel/586753
import os
import shutil
from os import listdir
from shutil import *
import numpy as np
from PIL import Image
import quadprog
import cvxopt
from cvxopt import  solvers
import matplotlib.pyplot as plt
class svm(object):

    def __init__(self):
        self.path = "orl_faces"
        self.error=[]
        self.c=100
        self.imgvec=[]
        self.imgvectest=[]
        self.weightofallclass={}
        self.classname = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16',
                 's17','s18','s19','s20','s21','s22','s23','s24','s25','s26','s27','s28','s29','s30',
                 's31','s32','s33','s34','s35','s36','s37','s38','s39','s40']


    def readfile(self):
        for directories in  (self.classname):
            path = os.path.join(self.path, directories)
            vectorslist = []
            self.totalimages = 200
            for dir in sorted(listdir(os.path.join(self.path, directories)))[:5]:
                filepath = os.path.join(path, dir)
                img = Image.open(filepath)
                arr = np.array(img,dtype='f')
                flat_arr = arr.flatten()
                self.imgvec.append(np.array(flat_arr))
                self.vectorlen=len(flat_arr)

    def train(self):
        #this index in used to change the y label to -1 for  class in each iteration
        ylabelindex=-1
        for cl in  (self.classname):
            ylabelindex=ylabelindex+1
            self.ylabels=[]
            for k in range(0,self.totalimages):
                self.ylabels.append(-1.0)

            #Changes the ylabels for class cl to 1
            for c in range(5*ylabelindex,5*ylabelindex+5):
                self.ylabels[c]=1.0
            ylabels = np.array(self.ylabels)


            imgvec=np.array(self.imgvec)
            X_Xtranspose = np.zeros((self.totalimages, self.totalimages),dtype='int')
            for i in range(self.totalimages):
                for j in range(self.totalimages):
                    X_Xtranspose[i, j] = np.dot((imgvec[i]),np.transpose(imgvec[j]))

            solvers.options['show_progress'] = False
#Calculate B
            b = cvxopt.matrix(0.0)

#Calculate P
            P = cvxopt.matrix(np.outer(ylabels,np.transpose(ylabels)) * X_Xtranspose)


#Calculate Q

            q = cvxopt.matrix(np.ones(self.totalimages) * -1)

#Calculate G

            gstdval = cvxopt.matrix(np.diag(np.ones

                                      (self.totalimages) * -1))

#Calculate H

            hstdval = cvxopt.matrix(np.zeros(self.totalimages))

#Calculate A
            A = cvxopt.matrix(ylabels, (1, self.totalimages), 'd')

            gslk = cvxopt.matrix(np.diag(np.ones(self.totalimages) * -1))

            G = cvxopt.matrix(np.vstack((gstdval, gslk)))
            hslk = cvxopt.matrix(np.ones(self.totalimages) * self.c)
            h = cvxopt.matrix(np.vstack((hstdval, hslk)))


            try:

                values = cvxopt.solvers.qp(P, q, G, h, A, b,)

                alpha = np.ravel(values['x'])

                supvec = alpha >1e-10

                self.supportvector_alpha = alpha[supvec]

                indicator = np.arange(len(alpha))[supvec]

                self.supvec_y = ylabels[supvec]

                self.image_sv = imgvec[supvec]

                self.w = np.zeros(self.vectorlen).reshape((1, self.vectorlen))

                for n in range(len(self.supportvector_alpha)):
                    hh = self.supportvector_alpha[n] * self.supvec_y[n]
                    hh = hh.reshape((1, 1))
                    self.w += hh * self.image_sv[n]


               # Intercept
                self.b = 0
                for n in range(len(self.supportvector_alpha)):
                    self.b += self.supvec_y[n]
                    self.b -= np.sum(self.supportvector_alpha* self.supvec_y * X_Xtranspose[indicator[n], supvec])

                self.b /= len(self.supportvector_alpha)

                weightandconstant={}
                weightandconstant['w']=self.w
                weightandconstant['b']=self.b
                self.weightofallclass[cl]=weightandconstant

            except Exception as exp:
                print(exp)

    def predict(self):
        for directories in (self.classname):

            path = os.path.join(self.path, directories)


            for dir in sorted(listdir(os.path.join(self.path, directories)))[5:]:
                filepath = os.path.join(path, dir)
                img = Image.open(filepath)
                arr = np.array(img, dtype='f')
                flat_arr = arr.flatten()
                predvalue = []
                predclass = []

                # calculated value for w and b for each class for
                #calculating one versur rest classification

                for cl, wandb in self.weightofallclass.iteritems():
                    predvalue.append(np.dot(wandb['w'],flat_arr)+wandb['b'])
                    predclass.append(cl)

            predval = max(predvalue)
            classnameindex = predvalue.index(predval)
            classname = predclass[classnameindex]
            actualclassname = directories

            if (classname == actualclassname):
                self.error.append(0)
            else:
                self.error.append(1)

#New Read file after switching trainig and test data

    def readfile1(self):
        for directories in  (self.classname):
            path = os.path.join(self.path, directories)
            vectorslist = []
            self.totalimages = 200
            for dir in sorted(listdir(os.path.join(self.path, directories)))[5:]:
                filepath = os.path.join(path, dir)
                img = Image.open(filepath)
                arr = np.array(img,dtype='f')
                flat_arr = arr.flatten()
                self.imgvec.append(np.array(flat_arr))
                self.vectorlen=len(flat_arr)

#prediction after  switching train and test
    def predict1(self):
        for directories in (self.classname):

            path = os.path.join(self.path, directories)


            for dir in sorted(listdir(os.path.join(self.path, directories)))[:5]:
                filepath = os.path.join(path, dir)
                img = Image.open(filepath)
                arr = np.array(img, dtype='f')
                flat_arr = arr.flatten()
                predvalue = []
                predclass = []

                # calculated value for w and b for each class for
                #calculating one versur rest classification

                for cl, wandb in self.weightofallclass.iteritems():
                    predvalue.append(np.dot(wandb['w'],flat_arr)+wandb['b'])
                    predclass.append(cl)

            predval = max(predvalue)
            classnameindex = predvalue.index(predval)
            classname = predclass[classnameindex]
            actualclassname = directories

            if (classname == actualclassname):
                self.error.append(0)
            else:
                self.error.append(1)

n = svm()
n.readfile()
n.train()
n.predict()
error=sum(n.error)/(len(n.error)*1.0)
print('accuracy',(1-error)*100)
acc=(1-error)*100

# Calculate accuracy aftre switching train and test data
n.readfile1()
n.train()

n.predict1()
error=sum(n.error)/(len(n.error)*1.0)
print('accuracy',(1-error)*100)
acc1=(1-error)*100
acc=(acc1+acc)/2
print('average accuracy',acc)








