# Main.py

import cv2
import numpy as np
import os

import pytesseract
from PIL import Image

import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


###################################################################################################
def main():
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training

    if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return  # and exit program
    # end if

    imgOriginalScene = cv2.imread("LicPlateImages/1.png")  # open image

    if imgOriginalScene is None:  # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return  # and exit pr10gram
    # end if

#this lisfOfPossiblePlates Contains 13 Plates
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates

    print(listOfPossiblePlates)

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates

    print(listOfPossiblePlates)

    cv2.imshow("imgOriginalScene", imgOriginalScene)  # show scene image

    if len(listOfPossiblePlates) == 0:  # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:  # else
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
        cv2.imshow("imgThresh", licPlate.imgThresh)
        # print(licPlate.imgThresh)
        # print(pytesseract.image_to_string(licPlate.imgThresh))
        # w, h = 512, 512
        # data = np.zeros((512, 512, 3), dtype=np.uint8)
        # data[0:256, 0:256] = [255, 0, 0]  # red patch in upper left
        # print(img)
        # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # imag = cv2.GaussianBlur(licPlate.imgPlate, (5, 5), 0)
        # imag1 = cv2.threshold(imgOriginalScene,127,255,cv2.THRESH_BINARY)
        # img = Image.fromarray(licPlate.imgPlate)

        imagag = cv2.bilateralFilter(licPlate.imgPlate, 9, 75, 75)
        img_grey = cv2.cvtColor(imagag, cv2.COLOR_BGR2GRAY)

        # imag1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        imag1 = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = Image.fromarray(imag1)
        # # img.save('my.png')
        img.show()
        pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract.exe'
        print(
            "licence plate possibility 1 = {0}".format(pytesseract.image_to_string(img, lang='eng', config='--psm 7')))

        if len(licPlate.strChars) == 0:  # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return  # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # draw red rectangle around plate

        print("\nlicense plate possibility 2 = " + licPlate.strChars + "\n")  # write license plate text to std out
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)  # write license plate text on the image

        cv2.imshow("imgOriginalScene", imgOriginalScene)  # re-show scene image

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)  # write image out to file

    # end if else

    cv2.waitKey(0)  # hold windows open until user presses a key

    return


# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect

    # for i, point in enumerate(p2fRectPoints):
    #     point = [int(point[0]), int(point[1])]
    #     p2fRectPoints[i] = point
    p2fRectPoints_temp = []
    for i, point in enumerate(p2fRectPoints[0:4]):
        point = [int(co_ord) for co_ord in point]
        p2fRectPoints_temp.append(point)

    cv2.line(imgOriginalScene, tuple(p2fRectPoints_temp[0]), tuple(p2fRectPoints_temp[1]), SCALAR_RED, 2)  # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints_temp[1]), tuple(p2fRectPoints_temp[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints_temp[2]), tuple(p2fRectPoints_temp[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints_temp[3]), tuple(p2fRectPoints_temp[0]), SCALAR_RED, 2)


# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0  # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0  # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0  # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))  # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  # call getTextSize

    # unpack roatated rect into center point, width and height, and angle
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)  # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)  # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # write the chars in below the plate
    else:  # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize  # unpack text size width and height

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))  # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))  # based on the text area center, width, and height

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


# end function

###################################################################################################
if __name__ == "__main__":
    main()
