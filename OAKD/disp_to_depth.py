import depthai as dai
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


point = (400, 300)


def show_distance(event, x, y, args, params):
    global point
    point = (x, y)


pipeline = dai.Pipeline()

# Define sources and outputs
stereo = pipeline.createStereoDepth()
stereo.setLeftRightCheck(False)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(True)

# Properties
monoLeft = pipeline.createMonoCamera()
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.out.link(stereo.left)

monoRight = pipeline.createMonoCamera()
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.out.link(stereo.right)

xoutDisp = pipeline.createXLinkOut()
xoutDisp.setStreamName("disparity")
stereo.disparity.link(xoutDisp.input)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(300, 300)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    disparityQueue = device.getOutputQueue(
        name="disparity", maxSize=1, blocking=True)
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=True)
    previewQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    calib = device.readCalibration()
    baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
    intrinsics = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.RIGHT, monoRight.getResolutionSize())
    focalLength = intrinsics[0][0]
    disp_levels = stereo.getMaxDisparity() / 95
    dispScaleFactor = baseline * focalLength * disp_levels

    # while True:
    #     dispFrame = np.array(disparityQueue.get().getFrame())
    #     with np.errstate(divide='ignore'):
    #         calcedDepth = (dispScaleFactor / dispFrame).astype(np.uint16)

    #     depthFrame = np.array(depthQueue.get().getFrame())

    #     # Note: SSIM calculation is quite slow.
    #     ssim_noise = ssim(depthFrame, calcedDepth)
    #     print(f'Similarity: {ssim_noise}')

    # Create mouse event
    cv2.namedWindow("Color frame")
    cv2.setMouseCallback("Color frame", show_distance)

    while True:

        dispFrame = np.array(disparityQueue.get().getFrame())

        with np.errstate(divide='ignore'):
            calcedDepth = (dispScaleFactor / dispFrame).astype(np.uint16)

        depthFrame = np.array(depthQueue.get().getFrame())

        colorFrame = previewQueue.get().getCvFrame()

        # depthFrame = depthQueue.get().getFrame()

        depthFrameColor = cv2.normalize(
            depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        # ret, depth_frame, color_frame = dc.get_frame()

        # Show distance for a specific point
        cv2.circle(colorFrame, point, 4, (0, 0, 255))
        distance = depthFrame[point[1], point[0]]

        cv2.putText(colorFrame, "{}mm".format(
            distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        cv2.imshow("depth frame", depthFrame)
        cv2.imshow("Color frame", colorFrame)
        key = cv2.waitKey(1)
        if key == 27:
            break
