import depthai as dai
import cv2
import numpy as np


def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()


def getMonoCamera(pipeline, isLeft):

    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)

    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    return mono


def getStereoPair(pipeline, monoLeft, monoRight):

    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo


def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


if __name__ == "__main__":

    mouseX = 0
    mouseY = 640

    pipeline = dai.Pipeline()

    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    # xoutDepth = pipeline.createXLinkOut()
    # xoutDepth.setStreamName("depth")

    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    # stereo.depth.link(xoutDepth.input)
    stereo.disparity.link(xoutDisp.input)
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)

    with dai.Device(pipeline) as device:

        disparityQueue = device.getOutputQueue(
            name="disparity", maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(
            name="rectifiedLeft", maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(
            name="rectifiedRight", maxSize=1, blocking=False)

        disparityMultiplier = 255/stereo.initialConfig.getMaxDisparity()

        cv2.namedWindow("Stereo Pair")
        cv2.setMouseCallback("Stereo Pair", mouseCallback)

        sideBySide = False

        while True:

            disparity = getFrame(disparityQueue)

            disparity = (disparity*disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

            leftFrame = getFrame(rectifiedLeftQueue)
            rightFrame = getFrame(rectifiedRightQueue)

            if sideBySide:
                imOut = np.hstack((leftFrame, rightFrame))
            else:
                imOut = np.uint8(leftFrame/2 + rightFrame/2)

            imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB)

            imOut = cv2.line(imOut, (mouseX, mouseY),
                             (1280, mouseY), (0, 0, 255), 2)
            imOut = cv2.circle(imOut, (mouseX, mouseY), 2, (255, 255, 128), 2)

            cv2.imshow("Stereo Pair", imOut)
            cv2.imshow("Disparity", disparity)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('t'):
                sideBySide = not sideBySide
