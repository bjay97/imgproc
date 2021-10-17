import cv2
import numpy as np
import scipy.signal
from datetime import datetime

def makeFrames(filePath,frameRate=1/30,Gray=False,resizeP=50,dbg=False):
    list_r = []
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()

        if hasFrames:
            scale_percent = resizeP  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            if Gray==True:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            list_r.append(resized)
        return hasFrames

    vidcap = cv2.VideoCapture(filePath)
    sec = 0
    # frameRate = 0.5 #//it will capture image in each 0.5 second
    #2fps
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
        if dbg==True:
            print('collecting frames' + str(count))

    cv2.destroyAllWindows()

    list_r = np.array(list_r,dtype='uint8')
    return list_r

def getDPs(image, threshold=.3,threshold2=.8,dbg=False):
    dark = np.count_nonzero(image <= 50)
    dark2 = np.count_nonzero((50<image) & (image<=100))
    light = np.count_nonzero((200 <= image))
    light2 = np.count_nonzero((150<=image) & (image<200))
    mid = np.count_nonzero((100 < image) & (image < 150))
    size = image.size
    darkP = round(dark / size, 2)
    darkP2 = round(dark2/size,2)
    lightP = round(light / size, 2)
    lightP2 = round(light2/size,2)
    midP = round(mid / size, 2)


    DP = (darkP, darkP2, midP, lightP,lightP2, size,threshold,threshold2)
    # if darkP > threshold:
    #     print('image too dark')
    # elif lightP > threshold:
    #     print('image too bright')
    # elif midP > threshold*normalMult:
    #     print('image too median')
    #     if darkP2 >= lightP2:
    #         print('image second quartile is lighter')
    #     else:
    #         print('image second quartile is darker')
    # else:
    #     print('just right')

    if dbg == True:
        print(f'darkP = {darkP}')
        print(f'midP = {midP}')
        print(f'lightP = {lightP}')
    return DP
def powTrans(img,pow):
  #low power is to brighten
  if pow > 0:
    c = 255/(np.max(img)**pow)
    img_pow = c*(img**pow)

  else:
    pow *= -1
    c = 255*(np.max(img)**pow)
    img_pow = c/(img**pow)
  img_pow = np.array(img_pow,dtype='uint8')
  return img_pow
def adjustDP_PowTrans(image,step=.5,threshold1=.2,threshold2=.8,itsLimit=6,dbg=False):
    DP = getDPs(image,threshold=threshold1,threshold2=threshold2,dbg=dbg)
    lStep = step
    dStep = 1 + step
    its = 0
    if dbg==True:
        print('adjusting Dps')

    while True:
        (darkP, darkP2, midP, lightP, lightP2, size, threshold, threshold2) = DP
        if its>itsLimit:
            break
        if darkP > threshold:
            if dbg == True:
                print('image too dark')
                print('brightening image')
            image = powTrans(image,lStep)
            DP = getDPs(image)
            its+=1
        elif lightP > threshold:
            if dbg == True:
                print('image too bright')
                print('darkening image')
            image = powTrans(image, dStep)
            DP = getDPs(image)
            its += 1
        elif midP > threshold2:
            if dbg == True:
                print('image too median')
            if darkP2 >= lightP2:
                if dbg == True:
                    print('image second quartile is lighter')
                image = powTrans(image, dStep)
                DP = getDPs(image)
                its += 1
            else:
                if dbg == True:
                    print('image second quartile is darker')
                image = powTrans(image, lStep)
                DP = getDPs(image)
                its += 1
        else:
            if dbg == True:
                print('just right')
                print(f'iterations:  {its}')
            break
    return image
def averaging_blur_custom(img,kernel,mode='same', boundary='fill', fillvalue=0,dbg=False):
    #mode = 'full', 'valid', 'same'
    #boundary = 'fill', 'wrap', 'symm'
    #fillvalue if boundary='fill'
    if kernel.shape[0] == kernel.shape[1]:
        kernel_c = kernel.shape[0]**2
        kernel = kernel/kernel_c
        if dbg==True:
            print(kernel)
    else:
        if dbg == True:
            print("Uneven kernel")
    img = scipy.signal.convolve2d(img,kernel,mode='same', boundary='fill', fillvalue=0)
    img = np.rint(img)
    img = np.array(img,dtype='uint8')
    if dbg == True:
        print(img)
    return img
def laplacian3(frame):
    kernel3 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    image = np.array(frame,dtype='int32')
    image = scipy.signal.convolve2d(image, kernel3,mode='same',boundary='fill',fillvalue=0)
    image = np.array(image, dtype='uint8')
    return image
def laplacian(frame):
    kernelArray = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    image = scipy.signal.convolve2d(frame,kernelArray)
    image = np.array(image,dtype='uint8')
    return image
def preprocScriptList(list,sKernelSize=3,dp = False,bl = False,lap = False,lap2=False,dgb=False):
    listr = []
    count = 0
    for i in range(0, len(list)):
        if sKernelSize >= 3 and not sKernelSize % 2 == 0:
            kernel = np.ones((sKernelSize, sKernelSize))

        image =list[i]
        if dp == True:
            image = adjustDP_PowTrans(image,dbg=dgb)
        if bl == True:
            image = averaging_blur_custom(image, kernel)
        if lap == True:
            image = laplacian3(image)
        if lap2 == True:
            image = laplacian(image)
        listr.append(image)
        count += 1
        print(count)
    listr = np.array(listr,dtype='uint8')
    return listr


def subBgAvList(list3d, threshold=25, dbg=False, batchsize=0):
    list_r = []
    if batchsize > 0:
        batches = int(list3d.shape[0] / batchsize)
        if dbg == True:
            print(f'batches: {batches}')
        for batchnumber in range(batches):
            start_index = batchnumber * batchsize
            end_index = start_index + batchsize
            if dbg == True:
                print(f'batch: {batchnumber}')
                print(f'indexes {start_index} to {end_index}')
            slice = list3d[start_index:end_index, :, :]
            naverage = np.average(slice, axis=0)
            naverage = np.array(naverage, dtype='int32')
            for i in range(start_index, end_index):
                frame = np.array(list3d[i], dtype='int32')
                diff = abs(frame - naverage)
                diff[diff <= threshold] = 0
                diff[diff > threshold] = 255
                diff = np.array(diff, dtype='uint8')
                list_r.append(diff)
                if dbg == True:
                    print(str(i) + 'frame Averaged')

    else:
        average = np.average(list3d, axis=0)
        average = np.array(average, dtype='int32')
        count = 0
        for i in range(list3d.shape[0]):
            frame = np.array(list3d[i], dtype='int32')
            diff = abs(frame - average)
            diff[diff <= threshold] = 0
            diff[diff > threshold] = 255
            diff = np.array(diff, dtype='uint8')
            list_r.append(diff)
            if dbg == True:
                print(str(count) + 'frame Averaged')
            count += 1

    list_r = np.array(list_r, dtype='uint8')
    return list_r

def morphologicsList(list,ksize1=3,ksize2=7):
    def dispersal_cleaner(array, ksize=3, clean=True):
        cleaner = np.ones((ksize, ksize))
        max_val = 255 * np.size(cleaner)
        img = np.array(array, dtype='int32')
        img2 = scipy.signal.convolve2d(img, cleaner, mode='same', boundary='fill', fillvalue=0)
        if clean == True:
            img2[img2 < max_val] = 0
            img2[img2 == max_val] = 255
        else:
            img2[img2 > 0] = 255

        img2 = np.array(img2, dtype='uint8')
        return img2

    listr = []
    count = 0
    for i in range(0, len(list)):
        array = list[i]
        array = dispersal_cleaner(array, ksize=ksize1, clean=True)
        array = dispersal_cleaner(array, ksize=ksize2, clean=False)
        listr.append(array)
        count += 1
        print(count)

    listr = np.array(listr, dtype='uint8')

    return listr
def fillerFrameV2List(list,dbg=True):
    listr = []
    count = 0
    for i in range(0, len(list)):
        a = np.nonzero(list[i])
        frame = np.copy(list[i])
        # Tuple of two arrays
        rows = a[0]
        cols = a[1]
        for i in cols:
            inds = np.where(cols == i)
            start = inds[0][0]
            end = inds[0][-1]

            frame[rows[start]:rows[end], i] = 255
        listr.append(frame)
        count += 1
        if dbg==True:
            print(f'Filler Frame {count} out of {len(list)}')

    listr = np.array(listr, dtype='uint8')
    return listr

def boundingbox3abv2(cords,array,crosshair=False,fillValue=100):
    center,left_edge,right_edge,top_edge,bottom_edge,success = cords
    if crosshair == True:
        array[center[0],left_edge[1]:right_edge[1]] = fillValue
        array[top_edge[0]:bottom_edge[0],center[1]] = fillValue

    #Top
    array[top_edge[0],left_edge[1]:right_edge[1]] = fillValue
    #Bottom
    array[bottom_edge[0],left_edge[1]:right_edge[1]] = fillValue
    #Left
    array[top_edge[0]:bottom_edge[0],left_edge[1]] = fillValue
    #right
    array[top_edge[0]:bottom_edge[0],right_edge[1]] = fillValue



    return array
def detectionScript(list1,list2,vertHi=.7,vertLo=.3,alt=False,crosshair=False,fillValue=(0,255,0),dbg=False):
    def boundingbox3av2(array, alt=False, vertHi=.7, vertLo=.3,dbg=False):
        # to find the index of maximum value along rows and cols, each.
        col_sum = np.sum(array, axis=0)
        row_sum = np.sum(array, axis=1)

        col_argmax = np.argmax(col_sum)
        row_argmax = np.argmax(row_sum)
        center = (row_argmax, col_argmax)
        if dbg==True:
            print(f'center: {center}')
        centerRowSlice = array[row_argmax, :]

        # New stuff for verticality
        diffCRS = np.diff(centerRowSlice)
        diff2 = np.nonzero(diffCRS)

        nright = np.where(diff2[0] > col_argmax)

        nnRight = diff2[0][nright]

        centerColSlice = array[:, col_argmax]

        CRSCounts = np.nonzero(centerRowSlice)
        if not CRSCounts[0].any() == 0:
            left = CRSCounts[0][0]
            right = CRSCounts[0][-1]

        else:
            left = 0
            right = 0

        CCSCounts = np.nonzero(centerColSlice)
        if not CCSCounts[0].any() == 0:
            top = CCSCounts[0][0]
            bottom = CCSCounts[0][-1]
        else:
            top = 0
            bottom = 0


        def calcVert():
            dLeft = abs(col_argmax - left)
            dRight = abs(right - col_argmax)
            dTop = abs(top - row_argmax)
            dBot = abs(row_argmax - bottom)
            dH = dLeft + dRight
            dV = dTop + dBot
            vert = dH / dV
            verT = dLeft, dRight, dTop, dBot, vert, dH, dV
            return verT

        vert = calcVert()
        print(vert)
        if vertLo <= vert[4] <= vertHi:
            if alt == True:
                if diff2[0].size > 1 and right < array.shape[1] - 1:
                    right = nnRight[0]

            left_edge = (row_argmax, left)
            right_edge = (row_argmax, right)
            top_edge = (top, col_argmax)
            bottom_edge = (bottom, col_argmax)
            success = True
        else:
            print('no vert')
            left_edge = (0, 0)
            right_edge = (0, 0)
            top_edge = (0, 0)
            bottom_edge = (0, 0)
            success = False
        cords = (center, left_edge, right_edge, top_edge, bottom_edge,success)
        return cords
    listr = []
    count = 0
    scount = 0
    for i in range(0, len(list1)):
        img1 = list1[i]
        img2 = list2[i]
        cords = boundingbox3av2(img1,alt=alt,vertHi=vertHi,vertLo=vertLo)
        if cords[5] == True:
            scount+=1
        frame = boundingbox3abv2(cords,img2,crosshair=crosshair,fillValue=fillValue)
        listr.append(frame)
        count += 1
        if dbg == True:
            print(count)
    percentage = scount / count
    if dbg == True:
        print(f'Scount: {scount}')
        print(f'Total Frames: {count}')
        print(f' Percentage success: {percentage}')

    listr = np.array(listr, dtype='uint8')
    return listr,percentage

if __name__ == '__main__':


    start_time = datetime.now()

    #inputLoc takes the input location of the video file.
    #Please make sure that the video has a fixed framerate ex: 30 fps instead of variable framerates.
    #Otherwise a bug in the codec runs an infinite frame collection script
    inputLoc = 'E:\\Recordings\\mob2quarter.mp4'

    framerate = 1/30 #1/15 seconds per frame i.e 15 fps, use fewer frames if running into memory restrictions

    resize_percentage = 75
    #100% of the frame size is preserved
    #if set to 50 will resize the frame to half of that of the original


    listg = makeFrames(inputLoc, frameRate=framerate, Gray=True, resizeP=resize_percentage)
    listc = makeFrames(inputLoc, frameRate=framerate, Gray=False, resizeP=resize_percentage)
    listp = preprocScriptList(listg, dp=True)
    lists = subBgAvList(listp, 60,batchsize=0)
    listM = morphologicsList(lists)
    listF = fillerFrameV2List(listM,dbg=True)
    listd, success = detectionScript(lists, listc, vertHi=.7, vertLo=.3)
    frame_width = listg.shape[1]
    frame_height = listg.shape[2]
    video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame_height, frame_width))
    for i in range(len(listd)):
        nFrame = listd[i]
        video.write(nFrame)
    video.release()
    print(f'success: {round(success, 2)}')

    end_time = datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1
    print("Execution Time: " + str(execution_time) + "seconds")
