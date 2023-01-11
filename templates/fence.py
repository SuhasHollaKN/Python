import json
import numpy as np
import pandas as pd
from PIL import Image, ImageColor
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from webcam import webcam
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import mediapipe as mp

# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet(r"D:\hackathon\poc\yolov3.weights", r"D:\hackathon\poc\yolov3.cfg")
# save all the names in file o the list classes
classes = []
with open(r"D:\hackathon\poc\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# get layers of the network
layer_names = net.getLayerNames()
# converts json file into dictionary
# Determine the output layer names from the YOLO model
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")


def load_json(file_name):
    with open(file_name, "r") as json_data:
        return json.loads(json_data.read())


def write_json(file_name, input_data):
    with open(file_name, "w") as outfile:
        outfile.write(json.dumps(input_data))


def filter_shapes(roi_json):
    rect_cordi = []
    poly_cordi = []
    circle_cordi = []
    # if len(roi_json)>2:
    for i in roi_json:
        if i['type'] == 'rect':
            rect_cordi.append(i)
        elif i['type'] == 'path':
            poly_cordi.append(i)
        elif i['type'] == 'circle':
            circle_cordi.append(i)
    return rect_cordi, poly_cordi, circle_cordi


canvas_result = None
options = st.sidebar.radio(
    "Select operations",
    ('ROI', 'Detection', 'Documentary'))
if options == "ROI":
    # st.write('<style>div.row-widget.stRadio > div{​flex-direction:row;justify-content: center;}​ </style>',
    # unsafe_allow_html=True)
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("rect", "circle", "transform", "polygon")
    )
    side_bar_name = "ROI Setting"
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    # fill_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    # capture image
    captured_image = webcam()
    if captured_image is None:
        st.write("Waiting for capture...")
    if captured_image is not None:
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(10, 255, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=captured_image if captured_image else None,
            update_streamlit=realtime_update,
            height=captured_image.height,
            width=captured_image.width,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        canvas_data = canvas_result.json_data["objects"][0]
        canvas_data["screenHeight"] = captured_image.height
        canvas_data["screenWidth"] = captured_image.width
        canvas_data["name"] = st.text_input(
            "Placeholder for the other text input widget",
            "This is a placeholder",
            key="placeholder",
        )
        write_json(r"roi_data.json", canvas_result.json_data["objects"])
        # Do something interesting with the image data and paths
        # if canvas_result.image_data is not None:
        #     st.image(canvas_result.image_data)
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(
                canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)
if options == "Detection":
    json_roi_data = load_json(r'roi_data.json')
    # json_roi_data_list = {​{​num: data['type']}​ for num, data in json_roi_data}​
    ROI_select = st.sidebar.multiselect('select type of ROI',
                                        ['rect', 'polygon'])
    rect, poly, circle = filter_shapes(json_roi_data)


    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if 'rect' in ROI_select:
                for roi_data in rect:
                    line_bbox_color = ImageColor.getcolor(roi_data["stroke"], "RGB")
                    img = cv2.rectangle(img, (roi_data['left'], roi_data['top']),
                                        (roi_data['left'] + roi_data['width'], roi_data['top'] + roi_data['height']),
                                        (line_bbox_color[2], line_bbox_color[1], line_bbox_color[0]), 2)
            if 'polygon' in ROI_select:
                for roi_data in poly:
                    line_polyline_color = ImageColor.getcolor(roi_data["stroke"], "RGB")
                    poly_line = [[(data[1]), (data[2])] for data in roi_data['path'] if len(data) > 2]
                    img = cv2.polylines(img, np.int32([poly_line]), 1,
                                        (line_polyline_color[2], line_polyline_color[1], line_polyline_color[0]))
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape
            # USing blob function of opencv to preprocess image
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0,
                                         swapRB=True, crop=False)
            # Detecting objects
            net.setInput(blob)
            outs = net.forward(output_layers)
            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            # We use NMS function in opencv to perform Non-maximum Suppression
            # we give it score threshold and nms threshold as arguments.
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = (0, 0, 0)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 2, color, 3)
                # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                #
                #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #
                #     img.flags.writeable = False
                #
                #     # Make detection
                #     result = pose.process(img)
                #
                #     # Recolor back to RGB
                #     img.flags.writeable = True
                #
                #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #
                #     # Render detection
                #     mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                #
                #     landmarks1 = result.pose_landmarks.landmark
                #
                #     left_index = [landmarks1[mp_pose.PoseLandmark.LEFT_INDEX.value].x * 640,
                #                   landmarks1[mp_pose.PoseLandmark.LEFT_INDEX.value].y * 480]
                #     right_index = [landmarks1[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * 640,
                #                    landmarks1[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * 480]
                #
                #     print("left hand point is: ", left_index)
                #     print("right hand point is: ", right_index)
                return img


    webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

< https: // teams.microsoft.com / l / message / 19: UUaL8IPEQbmOMOcnd5jBVACKwu2HYjq_wbDrDqkABec1 @ thread.tacv2 / 1673011389118?tenantId = 3980
f7fa - cefe - 4
bad - bc4d - e071a7870417 & amp;
groupId = 44760
a23 - d3eb - 4807 - aa9e - 9
c7f756bd7b0 & amp;
parentMessageId = 1673011389118 & amp;
teamName = ML
hackathon & amp;
channelName = General & amp;
createdTime = 1673011389118 & amp;
allowXTenantAccess = false >