import numpy as np
# import math
# import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# convert 2D matrix to patches or invert
def myimtocol(input1,rn1,rn2,n1,n2,tslide,xslide,f):
    if f == 1:
        n1,n2 = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        datasize = num1 * num2
        output1 = np.zeros((datasize, rn1, rn2), dtype='float32')

        for i in range(num2):
            for j in range(num1):
                if (i < num2 - 1):
                    if (j < num1 - 1):
                        output1[i * num1 + j, :, :] = input1[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2];
                    else:
                        output1[i * num1 + j, :, :] = input1[n1 - rn1:n1, i * xslide:i * xslide + rn2];
                else:
                    if (j < num1 - 1):
                        output1[i * num1 + j, :, :] = input1[j * tslide:j * tslide + rn1, n2 - rn2:n2];
                    else:
                        output1[i * num1 + j, :, :] = input1[n1 - rn1:n1, n2 - rn2:n2];
    else:
        [datasize, rn1, rn2] = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        output1 = np.zeros((n1,n2), dtype='float32')
        weight = np.zeros((n1,n2), dtype='float32')
        one = np.ones((rn1,rn2), dtype='float32')

        for i in range(num2):
            for j in range(num1):
                if (i < num2 - 1):
                    if (j < num1 - 1):
                        output1[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2] +=  np.squeeze(input1[i * num1 + j, :, :]);
                        weight[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2] += one;
                    else:
                        output1[n1 - rn1:n1, i * xslide:i * xslide + rn2] +=  np.squeeze(input1[i * num1 + j, :, :]);
                        weight[n1 - rn1:n1, i * xslide:i * xslide + rn2] += one;
                else:
                    if (j < num1 - 1):
                        output1[j * tslide:j * tslide + rn1, n2 - rn2:n2] +=  np.squeeze(input1[i * num1 + j, :, :]);
                        weight[j * tslide:j * tslide + rn1, n2 - rn2:n2] += one;
                    else:
                        output1[n1 - rn1:n1, n2 - rn2:n2] += np.squeeze(input1[i * num1 + j, :, :]);
                        weight[n1 - rn1:n1, n2 - rn2:n2] +=  one;

        output1 = output1/weight
    return output1
