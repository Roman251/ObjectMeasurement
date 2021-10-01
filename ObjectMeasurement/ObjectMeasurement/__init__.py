import cv2
import imutils

import numpy as np
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def order_points(pts):
  # sort the points based on their x-coordinates
  xSorted = pts[np.argsort(pts[:, 0]), :]
  # grab the left-most and right-most points from the sorted
  # x-roodinate points
  leftMost = xSorted[:2, :]
  rightMost = xSorted[2:, :]
  # now, sort the left-most coordinates according to their
  # y-coordinates so we can grab the top-left and bottom-left
  # points, respectively
  leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
  (tl, bl) = leftMost
  # now that we have the top-left coordinate, use it as an
  # anchor to calculate the Euclidean distance between the
  # top-left and right-most points; by the Pythagorean
  # theorem, the point with the largest distance will be
  # our bottom-right point
  D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
  (br, tr) = rightMost[np.argsort(D)[::-1], :]
  # return the coordinates in top-left, top-right,
  # bottom-right, and bottom-left order
  return np.array([tl, tr, br, bl], dtype="float32")

class obj_measure():
  def __init__(self, path:str) -> np.array:  
    self.image = cv2.imread(path) # will make changes to this image
    self.original = cv2.imread(path)
  
  # draw contours around the outer boundary in case it is needed but is not detected
  def draw_contour(self, thickness = 3, color = (255,0,0)) -> None:
    cv2.rectangle(self.image, (0, 0), (self.image.shape[1], self.image.shape[0]), color, thickness)

  def image_processing(self) -> None:
    # gray-scale
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # blur
    self.image = cv2.GaussianBlur(self.image, (5,5), 1)

    # canny-edge
    self.image = cv2.Canny(self.image, 50, 50)

    # dialate
    self.image = cv2.dilate(self.image, kernel=None, iterations=1)

    # erode
    self.image = cv2.erode(self.image, kernel=None, iterations=1)

  def contours(self, min_contour_area=100):

    """
    Will show the contours one by one on a black canvas that is the size of the image.
    Manually fetch the index of the contour where you first see the contour of the reference object

    min_contour_area : will discard the contour(will not be displayed) if the area of the contour is less than this number
    """
    
    black_image = np.zeros((self.image.shape[0],self.image.shape[1], 3)) # contours will be drawn on top of the black image
    cnts = cv2.findContours(self.image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # the parameters can be changed
    # cv2.RETR_TREE gives lots of contours 
    # cv2.RETR_EXTERNAL gives a lot less contours

    cnts = imutils.grab_contours(cnts)
 
    for (i,c) in enumerate (cnts):
      if cv2.contourArea(c) > min_contour_area:
         img = black_image.copy()
         cv2.drawContours(img, [c], -1, (255, 255, 255), 1)
         cv2.imshow(f"Contour Images-{i}", img)
         cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return cnts

  def calc_metrics_pixels(self, cnts, cnts_index:int, height=None, width=None) -> float:
    """
    Calculates Pixel-Length or Pixel-Width ratio
    """
    box = cv2.minAreaRect(cnts[cnts_index])

    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = order_points(box) # self.perspective.order_points(box)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br) 

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if width:
      pixelsPerMetric = dB / (width) # if width is known

    if height:
      pixelsPerMetric = dA / (height) # if the height is known

    return pixelsPerMetric

  def display_measurement(self, pixelsPerMetric, contours, minarea=100) -> None:
    for c in contours:
      # if the contour is not sufficiently large, ignore it
      if cv2.contourArea(c) < minarea:
        continue
      # compute the rotated bounding box of the contour
      orig = self.original.copy()
      box = cv2.minAreaRect(c)
      box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)

      box = np.array(box, dtype="int")
      # order the points in the contour such that they appear
      # in top-left, top-right, bottom-right, and bottom-left
      # order, then draw the outline of the rotated bounding
      # box
      box = order_points(box)
      cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
      # loop over the original points and draw them
      for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
      # unpack the ordered bounding box, then compute the midpoint
      # between the top-left and top-right coordinates, followed by
      # the midpoint between bottom-left and bottom-right coordinates
      (tl, tr, br, bl) = box
      (tltrX, tltrY) = midpoint(tl, tr)
      (blbrX, blbrY) = midpoint(bl, br)
      # compute the midpoint between the top-left and top-right points,
      # followed by the midpoint between the top-righ and bottom-right
      (tlblX, tlblY) = midpoint(tl, bl)
      (trbrX, trbrY) = midpoint(tr, br)
      # draw the midpoints on the image
      cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
      # draw lines between the midpoints
      cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
      cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)
      # compute the Euclidean distance between the midpoints
      dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY)) #length
      dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY)) # breadth
      # if the pixels per metric has not been initialized, then
      # compute it as the ratio of pixels to supplied metric
      # (in this case, cm)

      # compute the size of the object
      dimA = dA / pixelsPerMetric
      dimB = dB / pixelsPerMetric
      # draw the object sizes on the image
      cv2.putText(orig, "{:.2f}cm".format(dimA),
        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
      cv2.putText(orig, "{:.2f}cm".format(dimB),
        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
      # show the output image
      cv2.imshow("Object Measurement", orig)
      cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__=='__main__':
  
  image = obj_measure('../image/image.jpg')
  image.image_processing()

  contour = image.contours(min_contour_area=100)
  contour_index = 114 # fetched from the image title
  
  pixel_per_metric = image.calc_metrics_pixels(contour, contour_index , height=16.9, width=None)
  image.display_measurement(pixel_per_metric, contour, minarea=300)
