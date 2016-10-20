import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from itertools import cycle


parser=argparse.ArgumentParser()
parser.add_argument('-f','--file',help='file name of result',default='rst_0724.txt')
parser.add_argument('-r','--related',help='show related result',action='store_false')

args=parser.parse_args()
file_name=args.file

cycol = cycle('bgcmk').next

def data_from_txt(file_name):
    f = open(file_name)
    lines = f.readlines()
    data = []
    for l in lines:
        data.append([float(i) for i in l.split()])
    data = np.array(data)[:, [1, 0]].swapaxes(0, 1)
    return data
if __name__ == '__main__':
    data = data_from_txt(args.file)
    plt.plot(data[0], data[1], label="Our Model", color='r', linewidth=4)

    if args.related:
        for id,f_name in enumerate(os.listdir('./papers')):
            data=data_from_txt(os.path.join('papers',f_name))
            plt.plot(data[0], data[1], label=f_name, color=cycol(), linewidth=1)


    plt.xlabel("False positive")
    plt.ylabel("True positive rate")
    plt.ylim(.6,1)
    plt.xlim(-50,3000)
    plt.title("FDDB test result")
    plt.grid(True)
    plt.legend(loc=4)
    plt.show()
