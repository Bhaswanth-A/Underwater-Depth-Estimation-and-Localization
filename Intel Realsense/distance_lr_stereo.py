import cv2
import pyrealsense2 as rs
import numpy as np

points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
profile = pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        nir_lf_frame = frames.get_infrared_frame(1)
        nir_rg_frame = frames.get_infrared_frame(2)
        if not nir_lf_frame or not nir_rg_frame:
            continue
        nir_lf_image = np.asanyarray(nir_lf_frame.get_data())
        nir_rg_image = np.asanyarray(nir_rg_frame.get_data())
        # horizontal stack
        image = np.hstack((nir_lf_image, nir_rg_image))
        cv2.namedWindow('NIR images (left, right)', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IR Example', image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()


class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.points = rs.points()
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        nir_lf_frame = frames.get_infrared_frame(1)
        nir_rg_frame = frames.get_infrared_frame(2)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()


point = (400, 300)


def show_distance(event, x, y, args, params):
    global point
    point = (x, y)


# Initialize Camera Intel Realsense
dc = DepthCamera()

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

while True:
    ret, depth_frame, color_frame = dc.get_frame()

    # Show distance for a specific point
    cv2.circle(color_frame, point, 4, (0, 0, 255))
    distance = depth_frame[point[1], point[0]]

    cv2.putText(color_frame, "{}mm".format(
        distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
