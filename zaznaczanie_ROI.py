import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"
os.environ["PYTHONWARNINGS"] = "ignore"

import cv2
import supervision as sv

#ręczne zaznaczanie ROI(Region of interest) lewym przyciskiem myszy
def extract_roi_from_video(video_path, regions):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(region_name, img)

    num_rois = len(regions)
    ROIs = []
    print(f'Extracting ROIs from {video_path} with {num_rois} regions of interest')
    # regions = ['gazebo','mcg']
    for i in range(num_rois):
        region_name = regions[i]
        # create frame generator
        video_info = sv.VideoInfo.from_video_path(video_path)
        generator = sv.get_video_frames_generator(video_path)
        # acquire first video frame
        iterator = iter(generator)
        frame = next(iterator)
        # sv.plot_image(frame)

        # Create a window and set the callback function
        img = frame
        cv2.namedWindow(region_name)
        cv2.setMouseCallback(region_name, mouse_callback)

        points = []

        # region_name = input("Enter a name for the region - ")

        while True:
            cv2.imshow(region_name, img)

            # Wait for the user to press any key
            key = cv2.waitKey(1)  # & 0xFF
            if key == 27 or len(points) == 4:  # 'esc' key or 4 points selected
                break

        # Draw lines between the collected points
        if len(points) == 4:
            cv2.line(img, points[0], points[1], (0, 0, 255), 2)
            cv2.line(img, points[1], points[2], (0, 0, 255), 2)
            cv2.line(img, points[2], points[3], (0, 0, 255), 2)
            cv2.line(img, points[3], points[0], (0, 0, 255), 2)
            cv2.imshow(region_name, img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        for i in range(2):
            cv2.waitKey(1)

        # Return the coordinates and plot the frame with counter line
        # sv.plot_image(img)
        print("Selected Points:", points)

        # Extract the rectangular ROI based on the selected points
        roi_x = min(points, key=lambda x: x[0])[0]
        roi_y = min(points, key=lambda x: x[1])[1]
        roi_width = max(points, key=lambda x: x[0])[0] - roi_x
        roi_height = max(points, key=lambda x: x[1])[1] - roi_y

        # Extract ROI from the frame
        roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        x_range = [min(coord[0] for coord in points), max(coord[0] for coord in points)]
        y_range = [min(coord[1] for coord in points), max(coord[1] for coord in points)]

        # Adjust the range based on video width and height
        x_range_final = [max(x_range[0], 0), min(x_range[1], video_info.width - 1)]
        y_range_final = [max(y_range[0], 0), min(y_range[1], video_info.height - 1)]

        rectangle_range = [x_range_final, y_range_final]

        region = {"name": region_name,
                  "polygon": points,
                  "range": rectangle_range
                  }
        ROIs.append(region)

    return ROIs