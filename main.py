import cv2
import numpy as np
import scipy.ndimage as snd
import sys
global label
global mapImage


def mouse_click(event, x, y, flags, param):
    # to check if left mouse
    # button was clicked
    global label
    global mapImage
    if event == cv2.EVENT_LBUTTONUP:
        mask = label.reshape(output.shape[:-1])
        stateGroup = mask[(y, x)]

        # get a mask that's True at all the indices of state's color group
        stateGroupMask = mask == stateGroup
        stateGroupMask = np.asarray(stateGroupMask, dtype="uint8")

        blank = mapImage.copy()
        blank[stateGroupMask == 0] = 0

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9), (1, 1))

        stateGroupMaskClosed = cv2.morphologyEx(stateGroupMask, cv2.MORPH_CLOSE, element, iterations=5)

        # plot an image with only the state's cluster on a white background
        stateImage = mapImage.copy()

        # label all connected blobs in statemask
        bloblab = snd.label(stateGroupMaskClosed, structure=np.ones((3, 3)))[0]

        # create a mask for only the state where we clicked (including "closed" states/blobs
        onlyTheStateMask = bloblab == bloblab[y, x]

        # fills in the land bridges caused by the morph closing
        onlyTheStateMask[stateGroupMask == 0] = 0

        # apply mask to stateImage so we are left with just the state
        stateImage[onlyTheStateMask == 0] = 0

        # affine transformation for doubling size
        nz = onlyTheStateMask.nonzero()
        rows, cols = onlyTheStateMask.shape[:2]

        width = nz[0].max() - nz[0].min()
        height = nz[1].max() - nz[1].min()

        input_pts = np.float32([[nz[1].min(), nz[0].min()], [nz[1].min(), nz[0].max()], [nz[1].max(), nz[0].max()]])

        output_pts = np.float32([[nz[1].min() - .5 * height, nz[0].min() - .5 * width],
                                 [nz[1].min() - .5 * height, nz[0].max() + .5 * width],
                                 [nz[1].max() + .5 * height, nz[0].max() + .5 * width]])
        M = cv2.getAffineTransform(input_pts, output_pts)

        stateImage = cv2.warpAffine(stateImage, M, (cols, rows))
        onlyTheStateMask = np.asarray(onlyTheStateMask, dtype="uint8")
        onlyTheStateMask = cv2.warpAffine(onlyTheStateMask, M, (cols, rows))

        markedUpMapImage = mapImage.copy()
        markedUpMapImage[onlyTheStateMask != 0] = stateImage[onlyTheStateMask != 0]

        cv2.imshow("map", markedUpMapImage)

if __name__ == '__main__':
    global mapImage
    global label
    if sys.argv[1]:
        mapImage = cv2.imread(sys.argv[1])

        width = int(mapImage.shape[1] / 4)
        height = int(mapImage.shape[0] / 4)
        dim = (width, height)

        mapImage = cv2.copyMakeBorder(mapImage, 300, 600, 300, 300, cv2.BORDER_CONSTANT)

        Z = mapImage.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 2)

        K = 25
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        output = res.reshape(mapImage.shape)

        cv2.namedWindow("map", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("map", width + 650, height + 900)
        cv2.imshow("map", output)

        cv2.setMouseCallback("map", mouse_click)
        cv2.waitKey(0)
    else:
        print("Please pass in value for map image")

