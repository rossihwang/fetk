
import numpy as np 
import os 
import cv2
import matplotlib.pyplot as plt

def import_faces_as_cols(path):
    entries = os.scandir(path)
    entriesLst = list(entries)

    # Randomly read a image to get the information of the image
    img = cv2.imread(path+'/'+entriesLst[0].name, 0) # gray
    height, width = img.shape
    num = len(entriesLst)
    columns = np.zeros(((height*width), num))
    for i, e in enumerate(entriesLst):
        if e.is_file():
            img = cv2.imread(path+'/'+e.name, 0)
            columns[:, i, np.newaxis] = img.reshape(-1, 1)

    return columns, height, width 

def plot_cols_as_faces(cols, height, width):
    num = cols.shape[1]
    plt.figure()
    for i in range(num):
        face = cols[:, i].reshape(height, width)
        plt.subplot(num//5+1, 5, i+1)
        plt.imshow(face, cmap="gray", interpolation="bicubic")

def plot_end():
    plt.show()


def plot_scatter2d(X, y, title=None):
    f, ax = plt.subplots(figsize=(7, 5))
    if X.ndim > 2:
        print("dimension of X is larger that 2")
    xMin, xMax = np.percentile(X[:, 0], [0, 100])
    yMin, yMax = np.percentile(X[:, 1], [0, 100])
    numOfPerClass = np.bincount(y)
    numOfClass = numOfPerClass.size 
    if numOfClass > 7:
        print("Too many classes!")
    colors = "bgrcmyk"

    for i, pt in enumerate(X):
        ax.scatter(pt[0], pt[1], s=90, c=colors[y[i]])
    ax.set_ylim(yMin, yMax)
    ax.set_xlim(yMin, yMax)
    if title != None:
        ax.set_title(title)
    return f, ax


def iterate_imread(path):
    entries = os.scandir(path)
    samples = cv2.imread(entries.__next__().path, cv2.IMREAD_GRAYSCALE)

    for i in entries: 
        img = cv2.imread(i.path, cv2.IMREAD_GRAYSCALE)
        samples = np.dstack((samples, img))

    return samples

def main():
    # u = Uniform()
    # u.generate_table()
    s = iterate_imread("../CKP_Samples/anger")
    for i in range(10):
        plt.figure(i)
        plt.imshow(s[:,:,i], cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()
