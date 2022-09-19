import pywt
import numpy as np
import cv2 as cv

def waveletCompression(img, wavelet='haar', level=4, threshold=.05):
  coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
  coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
  Csort = np.sort(np.abs(coeff_arr.reshape(-1)))
  threshold = Csort[int(np.floor((1-threshold)*len(Csort)))]
  ind = np.abs(coeff_arr) > threshold
  Cfilt = coeff_arr * ind
  coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')
  return pywt.waverec2(coeffs_filt, wavelet=wavelet).astype('uint8')

def fourierCompression( img, threshold=.05 ):
  Bt = np.fft.fft2( img )
  Btsort = np.sort( np.abs( Bt.reshape( -1 ) ) )
  threshold = Btsort[ int( np.floor( ( 1-threshold )*len( Btsort ) ) ) ]
  ind = np.abs(Bt)>threshold          # Find small indices
  Atlow = Bt * ind                    # Threshold small indices
  return np.fft.ifft2( Atlow ).real   # Compressed image

def houghlines(img):
    edges = cv.Canny(img.numpy(), 50, 150, apertureSize = 3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    newImg = np.zeros((240, 426))
    for line in lines:
        print(line)
        x1,y1,x2,y2 = line[0]
        newImg = cv.line(newImg,(x1,y1),(x2,y2),(0,255,0),2)

    return newImg

# The clahe method is an adaptive algorithm from OpenCV 
# it divides the image into tiles and does the histogram equalization over the smaller area, it can amplify noise so it also limits the contrast. 
def clahe_method(img, limit=4, tile=(8,8)):
    clahe = cv.createCLAHE(clipLimit=float(limit), tileGridSize=tile)
    cla = clahe.apply(img)
    return cla

#still deveoping here
def orb_method(img1, img2, num_matches):
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    left_points = []
    right_points = []
    diffs = []
    for i in range(num_matches):
      #print("matches:")
      leftPt = kp1[matches[i].queryIdx].pt
      rightPt = kp2[matches[i].trainIdx].pt
      left_points.append(leftPt)
      right_points.append(rightPt)
      diff = (rightPt[0] - leftPt[0], rightPt[1]-leftPt[1])
      #print(leftPt, rightPt, diff, np.sqrt(diff[0]**2+diff[1]**2), np.rad2deg(np.arctan2(*diff)))
      diffs.append([np.sqrt(diff[0]**2+diff[1]**2), np.rad2deg(np.arctan2(*diff))])
    
    return diffs
    
def orb_method_arrow(img1, img2, num_matches):
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    left_points = []
    right_points = []
    for i in range(num_matches):
        #print("matches:")
        leftPt = kp1[matches[i].queryIdx].pt
        rightPt = kp2[matches[i].trainIdx].pt
        left_points.append(leftPt)
        right_points.append(rightPt)
        diff = (rightPt[0] - leftPt[0], rightPt[1]-leftPt[1])
        #print(leftPt, rightPt, diff, np.sqrt(diff[0]**2+diff[1]**2), np.rad2deg(np.arctan2(*diff)))
    
    blank = np.ones(img1.shape)
    #print(img1.shape)
    for i in range(num_matches):
        cv.arrowedLine(blank, pt1=tuple((int(left_points[i][0]), int(left_points[i][1]))), pt2=tuple((int(right_points[i][0]), int(right_points[i][1]))), color=(255,255,255), thickness=2)
    return blank