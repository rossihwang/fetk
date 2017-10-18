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
    Don't use
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
    """Generate a table for fast looking up uniform pattern
    Args:
        nBits: Use nBits to encode the pattern
        transThres: If pattern has at most [transThres] bitwise transitions, it will be classified as uniform pattern
    """
    def __init__(self, nBits=8, transThres=2):
        self.uniform = []
        self.nonuniform = []
        self.max = 2**nBits
        self.nBits = nBits 
        self.transThres = transThres
        self.init_lookup_table()
        self.nBins = len(self.uniform_idx) + 1

    def init_lookup_table(self):
        for i in range(self.max):
            if self.is_uniform(i):
                # print("{:08b}".format(i))
                self.uniform.append(i)
            else:
                self.nonuniform.append(i)

        # self.is_uniform(0b00000110)

    def is_uniform(self, data):
        bt = 0
        mask = 0x03
        d = data 
        for i in range(self.nBits-1):
            # print("{:08b}".format(d))
            if mask & d == 0x01:
                bt += 1
            d = d >> 1
        # print(bt)

        mask = 0x03
        d = data
        for i in range(self.nBits-1):
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
    """LBP feature class
    Generate LBP features for image or 

    cite:http://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    Args:
        nPoints: Number of coding bits in LBP feature
        radius: LBP radius
    """
    def __init__(self, nPoints, radius):
        self.nPoints = nPoints
        self.radius = radius
        self.pattern = LBPPattern(nPoints, 2)

    def describe(self, img, eps=1e-7):
        ## Compute the LBP representation of the image
        lbp = feature.local_binary_pattern(img, self.nPoints, self.radius, method="default")
        
        ## Compute the histogram
        (hist, _) = np.histogram(lbp.flatten(), bins=np.arange(0, 257), range=(0, 255))

        ## Compact the non-uniform pattern and put in last bin
        histCompact = np.zeros(len(self.pattern.uniform_idx)+1)
        histCompact[:-1] = hist[self.pattern.uniform_idx]
        histCompact[-1] = np.sum(hist[self.pattern.nonuniform_idx])

        ## Normalize the histogram
        # histCompact = histCompact.astype("float")
        # histCompact /= (histCompact.sum() + eps)
        histCompact /= histCompact.sum()

        return histCompact

    def describe_regions(self, img, regions, weights=None):
        """Given regions and weights, generate LBP feature
        Args:
            img: Input image.
            regions: tuple. Example (7, 6), 7 rows and 6 columns.
            weights: numpy array, it's size should equal to regions number.
        Returns: 
            numpy array, LBP histogram. 
        Raises:
            If weights size and regions number are not match, FeatureParameterInvalid() will be raised.
        """
        row, col = regions
        dh, dw = img.shape[0]/np.float(row), img.shape[1]/np.float(col)
        colTemp = np.ones(col); colTemp[0] = 0
        rowTemp = np.ones(row); rowTemp[0] = 0
        lx = np.uint8(np.add.accumulate(colTemp * dw))
        ly = np.uint8(np.add.accumulate(rowTemp * dh))
        dh = np.uint8(dh)
        dw = np.uint8(dw)
        hist = np.zeros((0, self.pattern.nBins))

        for y in ly:
            for x in lx:
                h = self.describe(img[y:y+dh, x:x+dw])
                hist = np.vstack((hist, h))
        if weights is not None:
            if regions[0]*regions[1] != weights.size:
                raise FeatureParameterInvalid("weights size doesn't match the regions size")
            else:
                hist *= weights
        ## Normalize
        hist /= np.sum(hist) 
        return hist.flatten()

    def plot_regions(self, img, regions):
        """Plot the regions
        """
        row, col = regions
        dh, dw = img.shape[0]/np.float(row), img.shape[1]/np.float(col)
        lx = np.uint8(np.add.accumulate(np.ones(col-1) * dw))
        ly = np.uint8(np.add.accumulate(np.ones(row-1) * dh))
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
        Args:
            nBins: Number of bins to be quantized to (from 10 to 80,    20-180, 40-360)
            maxAng: Set maxAng as 180 or 360 to set angel range to [0, 180] or [0, 360]
            level: Level of the pyramid, (limit the levels to 3 to prevent over fitting). 
                When level=0, PHOG=HOG. For level=3, there are actually 4 levels from
                0 to 3.
        '''
        if level > 3 or (maxAng not in [180, 360]):
            raise FeatureParameterInvalid()
        self.level = level
        self.nBins = nBins
        self.maxAng = maxAng
         
    def describe(self, img):
        '''
        compute PHOG. Row 1 is from level 0, row 2 to 5 are from level 1, and so on. 
        '''
        # step1: CANNY
        imgCnny = cv2.Canny(img, 100, 200) # TODO: need to optimize

        # step2: Sobel(3x3) -> gradient
        gX = cv2.Sobel(imgCnny, cv2.CV_64F, 1, 0, ksize=3)
        gY = cv2.Sobel(imgCnny, cv2.CV_64F, 0, 1, ksize=3)
        imgMag, imgAng = cv2.cartToPolar(gX, gY, angleInDegrees=True);
        
        # step3: Quantize orientation into K bins
        imgAngQ = self.quantize(imgAng, self.nBins, self.maxAng)

        # step4: Compute phog 
        phog = np.zeros((0, self.nBins))
        for lev in range(self.level+1):
            levelHog = self.compute_level_hog(imgMag, imgAngQ, lev)
            phog = np.vstack([phog, levelHog])
        
        ## Normalize
        phog = phog / np.sum(phog) 
        return phog.reshape(1, -1)

    def generate_weight(self, weights):
        '''
        weights: numpy array. For example, [0.1, 0.2, 0.3, 0.4], 0.1 is weight for level 0...
        '''
        if weights.size != self.level + 1:
            raise FeatureParameterInvalid("[PHOG]: weights size and level are not matched")
        n = np.arange(0, self.level+1)
        nRow = np.power(4, n)
        w = np.array([])
        for i, j in enumerate(nRow):
            w = np.append(w, np.ones(j) * weights[i])
        return w.T 

    def describe_with_weights(self, img, weights):
        phog = self.describe(img)
        w = self.generate_weight(weights)
        wPhog = phog * w 
        ## Normalize
        wPhog = wPhog / np.sum(wPhog)
        return wPhog.reshape(1, -1)

    def compute_level_hog(self, imgMag, imgAng, lev):
        """
        computer the HOG for the given level.
        lev: range 0~level
        """
        levelHog = np.zeros((0, self.nBins))
        dRow = imgAng.shape[0] // (2**lev)
        dCol = imgAng.shape[1] // (2**lev)
        dAng = self.maxAng // self.nBins # delta of angle
        b = np.arange(0, self.maxAng, dAng) # bins array for histogram
        for i in range(2**lev):
            for j in range(2**lev):
                regionMag = imgMag[i*dRow:(i+1)*dRow, j*dCol:(j+1)*dCol]
                regionAng = imgAng[i*dRow:(i+1)*dRow, j*dCol:(j+1)*dCol]
                hog = self.compute_hog(regionMag, regionAng, b)
                levelHog = np.vstack([levelHog, hog])
        return levelHog

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
        compute HOG. 
        '''
        hog = np.array([])
        for b in bins:
            hog = np.append(hog, np.sum(regionMag[regionAng==b])) # bin counts = magnitude * (counts of specific angle)
        return hog 
    
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
    img = np.arange(100).reshape(10, 10)
    weights = np.ones((5, 5)).reshape(-1, 1)

    lbp = LBP(8, 1)
    lbp.plot_regions(img, (5, 5))
    print(lbp.describe_regions(img, (5, 5), weights))
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
    phog = PHOG(nBins=20, maxAng=180, level=0)
    h1 = phog.describe_with_weights(imgTest, np.array([1]))
    expectH1 = np.array([ 0.2127,    0,    0,    0,
                             0,    0,    0, 0.2706,
                             0,    0, 0.3545,    0,
                             0,    0,    0, 0.0501,
                             0, 0.1121,    0,    0])
    hist = np.hstack([h1[0]])
    print(hist)
    print(expectH1)
    print("error: ", np.linalg.norm(hist - expectH1))
    
def lpq_unit_test():
    samp = np.load("../ckpAngerSample64x56.npy")
    lpq = LPQ(3)
    print(lpq.describe(samp[:,:,0]))

if __name__ == "__main__":
    # eigenface_test()
    lbp_test()
    # phog_unit_test()
    # lpq_unit_test()