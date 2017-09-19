import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt 
import cv2
from skimage import feature 
import scipy.signal as sig  
from scipy.spatial import distance as dist 

import sys
sys.path.append("../")
import matplotlib.lines as lines


class FeatureParameterInvalid(Exception):
    pass

class EigenFace():
    '''
    '''
    def __init__(self):
        super().__init__()
        self.isFit = False

    @property 
    def meanface(self):
        return self.mean

    def fit_transform(self, trainingCols, numOfComponents=None):
        ## Compute mean face
        self.numOfEigV = trainingCols.shape[1] - 1
        self.mean = np.mean(trainingCols, 1, keepdims=True)
        demean = trainingCols - self.mean
        ## Compute the eigenface
        scatter = np.dot(demean.T, demean)
        self.eigVal, eigVect = la.eig(scatter)
        self.eigVal = self.eigVal.real 
        eigVect = eigVect.real
        self.eigFace = np.dot(demean, eigVect) ## Use a trick 
        ## Normalize eigenFace, -> unit vector
        self.eigFace = np.divide(self.eigFace, la.norm(self.eigFace, axis=0))
        ## Sorting the eigenface and eigenvalue
        sortedIdx = np.argsort(self.eigVal)[::-1]
        self.eigVal = self.eigVal[sortedIdx]
        self.eigFace = self.eigFace[:, sortedIdx]
        self.isFit = True
        if numOfComponents == None:
            self.numOfComponents = self.numOfEigV
        else:
            self.numOfComponents = numOfComponents
        proj = self.eigFace[:, :self.numOfComponents].T @ demean
        return proj
    
    def transform(self, testCols, numOfComponents=None):
        if self.isFit != True:
            return 
        if numOfComponents==None:
            num = self.numOfComponents
        else:
            num = numOfComponents
        return self.eigFace[:, :num].T @ (testCols - self.mean)

    def num_of_eigv(self, varRatio):
        s = 0
        for n, v in enumerate(self.eigVal):
            s = s + v
            if s / np.sum(self.eigVal) > varRatio:
                return n+1
        
    def reconstruct(self, col, numOfComponents=None):
        '''
        This function can be used for face detection.
        '''
        if not self.isFit:
            print("Data is not trained")
            return 

        if numOfComponents == None:
            num = self.numOfEigV
        else:
            num = numOfComponents
        demean = col - self.mean
        proj = self.eigFace[:, :num].T @ demean
        return (self.eigFace[:, :num] @ proj) + self.mean

class FisherFace():
    '''
    Not validated
    '''
    def __init__(self):
        self.isFit = False

    def fit_transform(self, trainingCols, labels, numOfComponents=None):
        ## Compute PCA
        self.mean = np.mean(trainingCols, 1, keepdims=True)
        demean = trainingCols - self.mean 
        scatter = demean.T @ demean
        eigVal, eigVect = la.eig(scatter)
        eigVal = eigVal.real 
        eigVect = eigVect.real
        # print("eigVect: ", eigVect.dtype)
        eigVect= demean @ eigVect 
        sortedIdx = np.argsort(eigVal)[::-1]
        eigVect = eigVect[:, sortedIdx]
        ### Normalize eigenvectors
        eigVect = np.divide(eigVect, la.norm(eigVect, axis=0))
        ### 
        numOfClass = np.count_nonzero(np.bincount(labels)) # or np.bincount(labels).size
        numOfPerClass = np.zeros(numOfClass)
        for i in range(numOfClass):
            numOfPerClass[i] = labels[labels==i].size
         
        ## Compute between-class scatter
        meansOfClasses = np.zeros((trainingCols.shape[0], numOfClass))
        for i in range(numOfClass):
            meansOfClasses[:, i, np.newaxis] = np.mean(trainingCols[:, labels==i], 1, keepdims=True)
        ## Checked

        scatterBtw = np.zeros((trainingCols.shape[0], trainingCols.shape[0]))
        U = meansOfClasses - self.mean 
        scatterBtw = U @ np.diag(numOfPerClass) @ U.T  
        # for i in range(numOfClass):
        #     tmp = meansOfClasses[:, i, np.newaxis] - self.mean
        #     scatterBtw = scatterBtw + (np.dot(tmp, tmp.T))

        ## Compute within-class scatter
        scatterWth = np.zeros_like(scatterBtw)
        for i in range(numOfClass):
            X = trainingCols[:, labels==i] - meansOfClasses[:, i, np.newaxis]
            scatterWth = scatterWth + (X @ X.T) 
        # for i in range(numOfClass):
        #     tmp = trainingCols[:, labels==i] - meansOfClasses[:, i].reshape(-1, 1)
        #     scatterWth = scatterWth + np.dot(tmp, tmp.T)
        
        ## Sb, Sw diamensional reduction by PCA
        numOfEigV = trainingCols.shape[1] - numOfClass # N - c
        vect = eigVect[:, :numOfEigV]
        scatterBtwRdc = vect.T @ scatterBtw @ vect
        scatterWthRdc = vect.T @ scatterWth @ vect

        fshVal, fshVect = la.eig(la.inv(scatterWthRdc) @ scatterBtwRdc)
        ### fshVal, fshVect are complex here!
        fshVal = fshVal.real 
        fshVect = fshVect.real 
        sortedIdx = np.argsort(fshVal)[::-1]
        fshVal = fshVal[sortedIdx]
        print(fshVal)
        fshVect = fshVect[:, sortedIdx]  
        ### Normalize
        # fshVect = np.divide(fshVect, la.norm(fshVect, axis=0))
        
        ## fisherface
        self.numOfFshV = numOfClass - 1 # c - 1
        ## eigVect:(n)x(N-c), fshVect:(N-c)x(c-1) 
        self.fshFace = eigVect[:, 0:numOfEigV] @ fshVect[:, 0:self.numOfFshV]

        self.isFit = True
        if numOfComponents == None:
            self.numOfComponents = self.numOfFshV
        else:
            self.numOfComponents = numOfComponents
        return self.fshFace[:, :self.numOfComponents].T @ demean

    def transform(self, testCols, numOfComponents=None):
        if not self.isFit:
            return 
        if numOfComponents == None:
            num = self.numOfComponents
        else:
            num = numOfComponents
        return self.fshFace[:, :num].T @ (testCols - self.mean)

    # def reconstruct(self, col, numOfComponents=0):
    #     '''
    #     This function can be used for face detection.
    #     '''
    #     if not self.isFit:
    #         print("Data is not trained")
    #         return 

    #     if numOfComponents != 0:
    #         num = numOfComponents
    #     else:
    #         num = self.numOfFshV
    #     demean = col - self.mean
    #     proj = self.fshFace[:, :num].T @ demean
    #     return (self.fshFace[:, :num] @ proj) + self.mean
        

class LBPPattern():
    def __init__(self, bits=8, transThres=2):
        self.uniform = []
        self.nonuniform = []
        self.max = 2**bits
        self.bits = bits 
        self.transThres = transThres
        self.init_lookup_table()
        self.numOfBins = len(self.uniform_idx) + 1

    def init_lookup_table(self):
        for i in range(self.max):
            if self.is_uniform(i):
                # printprint("{:08b}".format(i))
                self.uniform.append(i)
            else:
                self.nonuniform.append(i)

        # self.is_uniform(0b00000110)

    def is_uniform(self, data):
        bt = 0
        mask = 0x03
        d = data 
        for i in range(self.bits-1):
            # print("{:08b}".format(d))
            if mask & d == 0x01:
                bt += 1
            d = d >> 1
        # print(bt)

        mask = 0x03
        d = data
        for i in range(self.bits-1):
            # print("{:08b}".format(d))
            if mask & d == 0x02:
                bt += 1
            d = d >> 1
        # print(bt)

        if bt <= self.transThres:
            return True 
        else:
            return False 
    @property
    def uniform_idx(self):
        return self.uniform

    @property
    def nonuniform_idx(self):
        return self.nonuniform

class LBP():
    '''
    cite:http://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    '''
    def __init__(self, numOfPoints, radius):
        self.numOfPoints = numOfPoints
        self.radius = radius
        self.pattern = LBPPattern(numOfPoints, 2)

    def describe(self, img, eps=1e-7):
        # Compute the LBP representation of the image, and 
        # then use the representation to build the histogram 
        # of patterns
        lbp = feature.local_binary_pattern(img, self.numOfPoints, self.radius, method="default")
        #print(lbp)
        
        (hist, _) = np.histogram(lbp.flatten(), bins=np.arange(0, 257), range=(0, 255))

        # Compact the non-uniform pattern in one bin
        histCompact = np.zeros(len(self.pattern.uniform_idx)+1)
        histCompact[:-1] = hist[self.pattern.uniform_idx]
        histCompact[-1] = np.sum(hist[self.pattern.nonuniform_idx])

        # Normalize the histogram
        histCompact = histCompact.astype("float")
        histCompact /= (histCompact.sum() + eps)

        # print(np.sum(histCompact))
        return histCompact

    def describe_regions(self, img, shape):
        '''
        output: [Region1Hist, Region2Hist, ..., RegionNHist].T
        '''
        row, col = shape
        dh, dw = img.shape[0]/np.float(row), img.shape[1]/np.float(col)
        colTemp = np.ones(col); colTemp[0] = 0
        rowTemp = np.ones(row); rowTemp[0] = 0
        lx = np.uint8(np.add.accumulate(colTemp * dw))
        ly = np.uint8(np.add.accumulate(rowTemp * dh))
        dh = np.uint8(dh)
        dw = np.uint8(dw)
        hist = np.zeros((0, self.pattern.numOfBins))

        for y in ly:
            for x in lx:
                h = self.describe(img[y:y+dh, x:x+dw])
                hist = np.vstack((hist, h))
        # print("[Debug] histogram shape: {}".format(hist.shape))
        return hist

    def plot_regions(self, img, shape):
        row, col = shape
        dh, dw = img.shape[0]/np.float(row), img.shape[1]/np.float(col)
        lx = np.uint8(np.add.accumulate(np.ones(col - 1) * dw))
        ly = np.uint8(np.add.accumulate(np.ones(row - 1) * dh))
        plt.figure()
        plt.imshow(img, cmap="gray")
        ax = plt.gca()
        ## Vertical lines
        for x in lx:
            ax.add_line(lines.Line2D([(x, x)], [(0, img.shape[0])], linewidth=2, color="red"))
        ## Horizonal lines
        for y in ly:
            ax.add_line(lines.Line2D([(0, img.shape[1])], [(y, y)], linewidth=2, color="red"))
        plt.plot()

# np.set_printoptions(threshold=np.nan)
class PHOG():
    def __init__(self, nBins, maxAng, level=3):
        '''
        level: level of the pyramid, (limit the levels to 3 to prevent over fitting). When level=1, PHOG=HOG
        bins: the number of bins to be quantized to (from 10 to 80, 20-180, 40-360)
        maxAng: set maxAng as 180 or 360 to set angel range to [0, 180] or [0, 360]
        '''
        if level > 3 or (maxAng not in [180, 360]):
            print(level)
            print(maxAng)
            raise FeatureParameterInvalid()
        self.level = level
        self.nBins = nBins
        self.maxAng = maxAng
         
    def describe(self, img):
        '''
        return:
        a python list will be returned. This list contains the histogram from all levels. 
        It is convenient for weighting when compute the distance. 
        '''
        # step1: CANNY
        imgCnny = cv2.Canny(img, 100, 200) # TODO: need to optimize
        # print("After Canny")
        # print(imgCnny)

        # step2: Sobel(3x3) -> gradient
        gX = cv2.Sobel(imgCnny, cv2.CV_64F, 1, 0, ksize=3)
        gY = cv2.Sobel(imgCnny, cv2.CV_64F, 0, 1, ksize=3)
        # imgAng = np.rad2deg(np.arctan(gY / gX)) # should be cautious that when gX is zero 
        # print("Gradient X, Y")
        # print(gX)
        # print(gY)
        imgMag, imgAng = cv2.cartToPolar(gX, gY, angleInDegrees=True);
        # print("Magnitude and angle image")
        # print(imgMag)
        # print(imgAng)
        
        # step3: Quantize orientation into K bins
        # print(imgOri)
        imgAngQ = self.quantize(imgAng, self.nBins, self.maxAng)
        # print("Angle after quantization")
        # print(imgAngQ)

        # step4: Compute pyrmaid histogram
        hist = []
        s = 0
        for i in range(self.level):
            h = self.pyramidHist(imgMag, imgAngQ, i)
            s = s + np.sum(h)
            hist.append(h)
        for i in range(self.level):
            hist[i] = hist[i] / s   
        return hist

    def pyramidHist(self, imgMag, imgAng, lev):
        """
        computer the pyramid histogram. 
        lev: range 0~(level-1)
        """
        hist = np.array([])
        dRow = imgAng.shape[0] // (2**lev)
        dCol = imgAng.shape[1] // (2**lev)
        for i in range(2**lev):
            for j in range(2**lev):
                regionMag = imgMag[i*dRow:(i+1)*dRow, j*dCol:(j+1)*dCol].flatten()
                regionAng = imgAng[i*dRow:(i+1)*dRow, j*dCol:(j+1)*dCol].flatten()
                dAng = self.maxAng // self.nBins # delta of angle
                b = np.arange(0, self.maxAng, dAng) # bins for histogram
                histTmp = self.compute_hog(regionMag, regionAng, b)
                hist = np.append(hist, histTmp)
        return hist

    @staticmethod
    def quantize(imgAng, nBins, maxAng):
        '''
        quantize the angles according to the given number of bins. 
        '''
        if maxAng == 180:
            imgAng[imgAng>=180] = imgAng[imgAng>=180] - 180
        dAng = maxAng // nBins
        for a in range(nBins+1):
            largerIdx = imgAng > (a*dAng)
            smallerIdx = imgAng < ((a+1)*dAng)
            idx = np.logical_and(largerIdx, smallerIdx)
            imgAng[idx] = a*dAng
        return imgAng

    @staticmethod
    def compute_hog(regionMag, regionAng, bins):
        '''
        compute the histogram of the angles, take the magnitude into consideration. 
        '''
        hist = np.array([])
        for b in bins:
            hist = np.append(hist, np.sum(regionMag[regionAng==b])) # bin counts = magnitude * (counts of specific angle)
        return hist 
    
# np.set_printoptions(precision=4) 
class LPQ():
    '''
    http://www.cse.oulu.fi/CMV/Downloads/LPQMatlab
    '''
    def __init__(self, rSize, alpha=None, rho=0.9):
        """
        Rsize: Region size, This is a square window(Rsize x Rsize) for the DFT
        alpha: First frequency point(except for DC). For example, a = 1 / rSize
        rho: coefficient for the covariance matrix
        """
        self.rSize = rSize
        if alpha == None:
            self.alpha = 1 / self.rSize
        else:
            self.alpha = alpha
        self.rho = rho 
        self.eps = 1e-15
        ## Compute some variables in advance
        ## Generate the covariance matrix, shape=(rSize**2)x(rSize**2)
        i, j = np.mgrid[:self.rSize, :self.rSize]
        pp = np.hstack([i.reshape(-1, 1), j.reshape(-1, 1)])
        dd = dist.squareform(dist.pdist(pp))
        covMt = np.power(self.rho, dd)
        # print(covMt)
        ## Generate the transformation matrix
        ## kernel of DFT, n is the index in spatial domain(e.g. Constant), k is the index in frequency domain(e.g. Constant/N)
        w = lambda n,k: np.exp(-1j*2*np.pi*n*k) 
        r = (self.rSize-1) / 2
        spIdx = np.arange(-r, r+1).reshape(1, -1)
        self.w0 = w(spIdx, 0) 
        self.w1 = w(spIdx, self.alpha)
        self.w2 = np.conjugate(self.w1) # w(spIdx, -self.alpha)   
        # print("w0:{}\nw1:{}\nw2:{}\n".format(self.w0, self.w1, self.w2)) # Checked
        ## Compute W ~ transMt
        q1 = self.w0.T @ self.w1
        q2 = self.w1.T @ self.w0
        q3 = self.w1.T @ self.w1
        q4 = self.w1.T @ self.w2 
        # print("q1:{}\nq2:{}\nq3:{}\nq4:{}\n".format(q1, q2, q3, q4)) # CHecked

        transMt = np.vstack([q1.real.flatten(), 
                             q1.imag.flatten(), 
                             q2.real.flatten(), 
                             q2.imag.flatten(), 
                             q3.real.flatten(), 
                             q3.imag.flatten(), 
                             q4.real.flatten(), 
                             q4.imag.flatten()])
        # print(transMt)
        ## Compute the covariance matrix of the transform coefficient, D
        covTransMt = transMt @ covMt @ transMt.T  
        # print(covTransMt.shape)
        ## Use "random" (almost unit) diagonal matrix to avoid multiple eigenvalues, refer to original matlab code.
        A = np.diag([1.000007, 1.000006, 1.000005, 1.000004, 1.000003, 1.000002, 1.000001, 1])
        ## SVD
        ada = A @ covTransMt @ A
        ada[np.abs(ada)<self.eps] = .0 # If skip this step, Fx would be different for signs of some columns 
        U, S, Vh = la.svd(ada)
        self.V = Vh.T
        # self.V[np.abs(self.V)<1e-5] = .0
        # print("V", self.V)

    def describe(self, img):
        convMode = "valid"
        # step1: Convolv with four filters(four 2d frequencies)
        # step2: Take out four complex coefficients
        Fu1 = sig.convolve2d(sig.convolve2d(img, self.w0.T, convMode), self.w1, convMode)  
        Fx = np.zeros((Fu1.shape[0], Fu1.shape[1], 8))
        Fx[:, :, 0] = Fu1.real
        Fx[:, :, 1] = Fu1.imag
        Fu2 = sig.convolve2d(sig.convolve2d(img, self.w1.T, convMode), self.w0, convMode)
        Fx[:, :, 2] = Fu2.real
        Fx[:, :, 3] = Fu2.imag
        Fu3 = sig.convolve2d(sig.convolve2d(img, self.w1.T, convMode), self.w1, convMode)
        Fx[:, :, 4] = Fu3.real
        Fx[:, :, 5] = Fu3.imag
        Fu4 = sig.convolve2d(sig.convolve2d(img, self.w1.T, convMode), self.w2, convMode)
        Fx[:, :, 6] = Fu4.real 
        Fx[:, :, 7] = Fu4.imag
        # print("Fu1:{}\nFu2:{}\nFu3:{}\nFu4:{}\n".format(Fu1, Fu2, Fu3, Fu4)) # Fu3, Fu4 is wrong
        FxRow, FxCol, FxNum = Fx.shape
        Fx = Fx.reshape(FxRow*FxCol, FxNum)  
        
        # Fx[Fx<self.eps] = .0
        # print("Fx:", Fx)
        # print("V", self.V)
        # print("Fx", Fx)          
        ## step3: Whitening transform
        ## FIXME the result is a little bit different from the original MATLAB code
        Gx = Fx @ self.V
        # Gx[np.abs(Gx)<1e-15] = .0
        # print("Gx:", Gx)
        Gx = Gx.T.reshape(FxNum, FxRow, FxCol)
        
        # print("Gx0", Gx)
        # print("Gx1", Gx[:,:,1])
        # print("Gx6", Gx[:,:,6])
        # print("Gx7", Gx[:,:,7])
        # step4: Quantization
        Bx = np.zeros((FxRow, FxCol))
        for i in range(FxNum):
            Bx = Bx + np.double(Gx[i, :, :] >= 0) * (2**(i))
        # print(Bx)
        Bx=np.uint8(Bx)
        # print(np.bincount(Bx.flatten()))
        # step5: Compute histogram for Bx
        return np.histogram(Bx.flatten(), bins=np.arange(0, 257), range=(0, 255))
            
def lbp_test():
    img = cv2.imread("../training/anger/S011_004_00000021.png", cv2.IMREAD_GRAYSCALE)
    # img = np.zeros((3, 3))
    lbp = LBP(8, 1)
    # lbp.plot_regions(img, 4, 4)
    print(lbp.describe_regions(img, 4, 4))
    # histLBP = lbp.describe(img)
    # plt.figure()
    # plt.stem(histLBP)
    plt.show() 

def phog_unit_test():
    '''
    Part of these have checked manually, exclude(canny, gradient).
    for the gradient, I am not sure how opencv do the padding, but
    I am sure that it use these two kernel
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    '''
    imgTest = np.array([[  8, 226,  99,  15],
                        [204,  50, 249,  65],
                        [ 65,  60,  45, 248],
                        [ 71, 170, 167,  70]], dtype=np.uint8)
    print(imgTest)
    phog = PHOG(nBins=20, maxAng=180, level=1)
    h1 = phog.describe(imgTest)
    expectH1 = np.array([ 0.2127,    0,    0,    0,
                             0,    0,    0, 0.2706,
                             0,    0, 0.3545,    0,
                             0,    0,    0, 0.0501,
                             0, 0.1121,    0,    0])
    hist = np.hstack([h1[0]])
    print(hist, expectH1)
    print("error: ", np.linalg.norm(hist - expectH1))

def phog_test():
    img = cv2.imread("/home/rossihwang/faceRecognition/my/101_ObjectCategories/accordion/image_0001.jpg", cv2.IMREAD_GRAYSCALE)
    phog = PHOG(bins=20, maxAng=180, level=3)
    h1 = phog.describe(img)
    for i in range(3):
        print(len(h1[i]))
    
def lpq_unit_test():
    samp = np.load("../ckpAngerSample64x56.npy")
    lpq = LPQ(3)
    print(lpq.describe(samp[:,:,0]))

if __name__ == "__main__":
    # eigenface_test()
    # __lbp_test()
    phog_unit_test()
    # lpq_unit_test()