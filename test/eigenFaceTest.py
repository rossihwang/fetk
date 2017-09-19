import sys
sys.path.append("../")
from feature import EigenFace
import utils as utl
import numpy as np

def main():
    trainingSamp = np.load("../ckpAngerSample64x56.npy")
    row, col, num = trainingSamp.shape
    trainingSamp = trainingSamp.reshape(row*col, num)
    print(trainingSamp.shape)
    ef = EigenFace()
    proj = ef.fit_transform(trainingSamp, 5)
    print(proj.shape)
    reconSamp = ef.reconstruct(trainingSamp, 15)
    utl.plot_cols_as_faces(reconSamp, row, col)
    utl.plot_end()

if __name__ == "__main__":
    main()
