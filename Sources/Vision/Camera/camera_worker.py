
import sys
from os import path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

current_dir = path.dirname(path.abspath(__file__))
while path.split(current_dir)[-1] != r'Heron':
    current_dir = path.dirname(current_dir)
sys.path.insert(0, path.dirname(current_dir))

import cv2 as cv2
from Heron import general_utils as gu
from Heron.gui.visualisation_dpg import VisualisationDPG

acquiring_on = False
capture: cv2.VideoCapture
vis: VisualisationDPG
frame_index = 0
time_stamp: str
ts_frame_index: bool
ts_font_size: int
font_file = path.join(current_dir, 'resources', 'fonts', 'SF-Pro-Rounded-Regular.ttf')


def now():

    hour = datetime.now().hour
    minute = datetime.now().minute
    second = datetime.now().second
    micro = datetime.now().microsecond

    string_time = '{}:{}:{}:{}'.format(hour, minute, second, micro)

    hms_state = hour * 60 * 60 + minute * 60 + second
    hms_pixels = gu.base10_to_base256(hms_state, normalise=True)
    micro_pixels = gu.base10_to_base256(micro, normalise=True)

    return string_time, hms_pixels, micro_pixels


def initialise(worker_object):
    global capture
    global acquiring_on
    global vis
    global frame_index
    global time_stamp
    global ts_frame_index
    global ts_font_size

    try:
        visualisation_on = worker_object.parameters[0]
        cam_index = worker_object.parameters[1]
        time_stamp = worker_object.parameters[3]
        ts_frame_index = worker_object.parameters[4]
        ts_font_size = worker_object.parameters[5]
        resolution_x = int(worker_object.parameters[2].split('x')[0])
        resolution_y = int(worker_object.parameters[2].split('x')[1])

        capture = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)

        acquiring_on = True

        worker_object.savenodestate_create_parameters_df(visualisation_on=visualisation_on, camera_index=cam_index,
                                                         resolution=(resolution_x, resolution_y),
                                                         time_stamp=time_stamp, ts_frame_index=ts_frame_index,
                                                         ts_font_size=ts_font_size)

        vis = VisualisationDPG(_node_name=worker_object.node_name, _node_index=worker_object.node_index,
                               _visualisation_type='Image', _buffer=1)

        print('Got camera parameters. Starting capture')
    except:
        gu.accurate_delay(1)
        acquiring_on = False

    return acquiring_on


def add_timestamp(frame):
    global frame_index
    global time_stamp
    global ts_frame_index
    global ts_font_size
    global font_file

    font = ImageFont.truetype(font_file, ts_font_size)
    coordinates = (10, 5 + ts_font_size)

    match time_stamp:
        case 'Top Right':
            coordinates = (frame.shape[1] - 11 * ts_font_size, 5 + ts_font_size)
        case 'Bottom Left':
            coordinates = (10, frame.shape[0] - (5 + ts_font_size))
        case 'Bottom Right':
            coordinates = (frame.shape[1] - 11*ts_font_size, frame.shape[0] - (5 + ts_font_size))

    str_datetime, hms_pixels, micro_pixels = now()

    frame[:3, 0, 0] = hms_pixels
    frame[:3, 1, 0] = micro_pixels

    if ts_frame_index:
        str_datetime = '{}, i={}'.format(str_datetime, frame_index)

    pil_image = Image.fromarray(frame)
    draw_image = ImageDraw.Draw(pil_image)
    draw_image.text(coordinates, str_datetime, (255, 255, 255), font)

    return np.array(pil_image)


def run_camera(worker_object):
    global capture
    global acquiring_on
    global vis
    global frame_index
    global time_stamp

    while not acquiring_on:
        gu.accurate_delay(1)

    while acquiring_on:
        ret, result = capture.read()
        if ret:
            worker_object.savenodestate_update_substate_df(frame=frame_index)
            frame_index += 1

            if time_stamp != 'No':
                result = add_timestamp(result)
            worker_object.send_data_to_com(result)

            vis.visualisation_on = worker_object.parameters[0]
            vis.visualise(result)


def on_end_of_life():
    global capture
    global acquiring_on
    global vis

    acquiring_on = False
    try:
        capture.release()
        vis.end_of_life()
    except:
        pass


if __name__ == "__main__":
    gu.start_the_source_worker_process(initialisation_function=initialise,
                                       work_function=run_camera,
                                       end_of_life_function=on_end_of_life)