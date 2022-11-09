#!/usr/bin/env python3
import os, numpy as np, cv2, recognition as rec

filename = ""
training = False
restart = True


def output(name, image):
    # A simple procedure to output generated images to the user where required
    # cv2.imshow outputs the image to the user with a title, waitKey waits until
    # the user interacts with the program, before closing any display windows

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def binary_thresholding(img, thresh):
    # Recives an input image, and the type of thresholding to be performed.
    # the image is initially converted from colour into a greyscale image,
    # and the image is slightly blurred as to reduce unnecessary detail within
    # the image, things such as noise.
    # The user can opt to binary threshold the image, and can change the
    # threshold value manually within the first cv2.threshold statement, index 1.
    # More often than not however, the program will use the pre-defined Otsu
    # threshold function, which will automatically determine a suitable threshold
    # value for the image, which theoretically should perform better than a manual
    # global threshold value.
    # Finally, morphological operators are applied to the image. These operators
    # are used to once again remove undesirable features from the image. They do so
    # by shrinking all white regions (foreground elements), so that any small details
    # are erased from the image, before expanding the remaining white regions back to
    # their original size.
    # Due to interference such as shadows, some contours that should be whole might end
    # up getting split. To rejoin these contours, we expand the remaining contours past
    # their original size, as to connect those contours, before returning them back to
    # their original size.
    # The output is a binarised image, post morphological processing, with background
    # items in colour black, and foreground items in colour white.

    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey
    greyscale = cv2.GaussianBlur(greyscale, (5, 5), 0)  # apply a blur

    if thresh == 'binary':
        _, binary = cv2.threshold(greyscale, 200, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(greyscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    # binary_thresh = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_thresh = cv2.erode(binary, kernel, iterations=2)
    binary_thresh = cv2.dilate(binary_thresh, kernel, iterations=2)

    binary_thresh = cv2.dilate(binary_thresh, kernel, iterations=5)
    binary_thresh = cv2.erode(binary_thresh, kernel, iterations=5)

    return binary_thresh


def colour_thresholding(img):
    # Similarly to binary thresholding, used to find extract 'useful' features from
    # an image. Instead of using a grey-level as a threshold value, we instead use
    # colour masks, as to extract all pixels that lie within specified value ranges
    # This project uses masks for white, orange and red, as these colours are commonly
    # used on aircraft liveries. Any details with these colours are extracted from the
    # original image, before being drawn onto a new image colour_thresh, which is then
    # passed to the binary thresholding function, so that it can be binarised for
    # contouring.

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_white = cv2.inRange(img, np.array([174, 184, 193]), np.array([255, 255, 255]))
    mask_orange = cv2.inRange(img, np.array([191, 124, 85]), np.array([239, 205, 159]))
    mask_red = cv2.inRange(img, np.array([122, 48, 44]), np.array([230, 115, 127]))

    mask_temp = cv2.bitwise_or(mask_white, mask_orange)
    mask = cv2.bitwise_or(mask_temp, mask_red)
    colour_thresh = cv2.bitwise_and(img, img, mask=mask)

    binary_colour_thresh = binary_thresholding(colour_thresh, 'otsu')

    return binary_colour_thresh


def contouring(img, thresh_img):
    # probably the most complex function within this file. The function
    # reads in the original image, and the binarised colour threshold image.
    # It first creates a copy of the original image that can be manipulated.
    # Then it creates a blank canvas as to draw any suitable contours onto
    # said canvas.
    # These contours will be the outlines of any features found during the
    # thresholding stage. We dont care about any contours inside of other
    # contours, hence RETR_EXTERNAL, it finds outer-most contours only.
    # These contours are then analysed, ensuring they are of sufficient size,
    # before being drawn onto our canvas. As well as drawing the contours, we
    # draw the a minimum area bounding box around the contour, which will be
    # passed to the function extraction, which fixes perspective warp and extracts
    # the contour into a standardised 64x64 pixel image. If the model isn't training,
    # this extract is passed to recognition.py, where the model attempts to classify
    # the extract. The classifier returns a label, which is written onto the output
    # image alongside the corresponding bounding box.

    contoured = img.copy()
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(thresh_img.shape, np.uint8)
    mask.fill(255)
    counter = 0
    for contour in contours:
        if cv2.contourArea(contour) > 400:
            cv2.drawContours(mask, contour, -1, (0, 0, 0), 1)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(contoured, [box], 0, (0, 0, 0), 2)
            extract = contour_extraction(img, box, counter)
            if not training:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) == 4:
                    label = 'building'
                else:
                    label = rec.classify(extract)
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(contoured, text=label, org=(cx, cy),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),
                            thickness=2, lineType=cv2.LINE_AA)

            counter += 1
    contoured = cv2.bitwise_and(contoured, contoured, mask=mask)
    if not training:
        output("Found aircraft", contoured)

    return contoured


def contour_extraction(img, contour, counter):
    global model, classes
    # note: similar to my own ce-316 computer vision assignment submission

    # code here is influenced by the link below and the UoE CE316
    # Computer Vision lecture notes.
    # First off we find the pixel values of the corners of the box
    # contour, then we declare two arrays, storing the corners of
    # the box contours found, and the new positions the corners will
    # reside in. We then call cv2.getPerspectiveTransform to find the
    # transformation maxtrix, and cv2.warpPerspective to perform the
    # allignment correction in the new 64x64pixel image.
    # https://pyimagesearch.com/2016/02/08/opencv-shape-detection/
    # taking the found contours and extracting them out into
    # their own images.

    global filename

    try:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        approx = np.array(approx)
        approx = np.reshape(approx, (4, 2))

        pos = np.zeros((4, 2), dtype=np.float32)
        pos[0, :] = approx[1]  # ULx, ULy
        pos[1, :] = approx[2]  # LLx, LLy
        pos[2, :] = approx[3]  # LRx, LRy
        pos[3, :] = approx[0]  # URx, URy

        n = 64
        opos = np.zeros((4, 2), dtype=np.float32)

        opos[0, :] = [0, 0]
        opos[1, :] = [0, n - 1]
        opos[2, :] = [n - 1, n - 1]
        opos[3, :] = [n - 1, 0]

        xform = cv2.getPerspectiveTransform(pos, opos)
        warp = cv2.warpPerspective(img, xform, (n, n))
        write_to_file(warp, counter)
        return warp

    except ValueError:
        print("Contour skipped; Value Error")


def write_to_file(image, counter):
    # checks to see if the desired output files exist, if not creates them.
    # The new images are renamed, as to accurately inform the user of their
    # content and source image. Upon restarting the program, the directories
    # in question are wiped so that the data is fresh for each program execution.

    global training, restart
    if training:
        path = 'training/extracts_training/'
    else:
        path = 'testing/extracts_testing'
    if not os.path.exists(path):
        os.makedirs(path)
    image_name = filename.split('.')
    new_file = 'extract_' + image_name[0] + "_" + str(counter) + '.png'
    cv2.imwrite(os.path.join(path, new_file), image)
    restart = False


def main_seg(directory, train):
    # instead of being object oriented, the program is procedural, and so this
    # is a replacement for a source __init.py__ file. It connects the functions
    # together as to ensure the program can execute.

    global filename, training
    training = train
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        image = cv2.imread(directory + "/" + filename)

        # binary_thresh = binary_thresholding(image, 'otsu') # colour threshold instead
        binary_colour_thresh = colour_thresholding(image)
        contoured_image = contouring(image, binary_colour_thresh)
