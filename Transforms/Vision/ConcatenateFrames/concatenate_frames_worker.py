
import sys
from os import path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageDraw
import copy

current_dir = path.dirname(path.abspath(__file__))
while path.split(current_dir)[-1] != r'Heron':
    current_dir = path.dirname(current_dir)
sys.path.insert(0, path.dirname(current_dir))

from Heron.communication.socket_for_serialization import Socket
from Heron import general_utils as gu
from Heron.Operations.Transforms.Vision.CvtColor import cvtColor_com
from Heron.gui.visualisation_dpg import VisualisationDPG

vis: VisualisationDPG
cameras_info: dict
cameras_setup_string: str
image_depth: int
pixel_gap: int
received_one_frame_from_all_cams: False
result_frame: np.ndarray
widths = []
heights = []


def setup_output_frame():
    global cameras_setup_string
    global cameras_info
    global result_frame
    global pixel_gap
    global image_depth
    global widths
    global heights

    cameras_info = {}
    cameras_setup_list = cameras_setup_string.split(', ')
    for c in cameras_setup_list:
        index = 'Camera##{}'.format(c.split(':')[0])
        info = {'Resolution': [int(i) for i in c.split(':')[1].split('x')],
                'Position': [int(i) for i in c.split(':')[2].split('x')]}
        #info['Data'] = np.zeros((info['Resolution'][0], info['Resolution'][1], image_depth))
        cameras_info[index] = info

    widths = np.zeros(np.max([cameras_info[c]['Position'][0] for c in cameras_info]) + 1)
    heights = np.zeros(np.max([cameras_info[c]['Position'][1] for c in cameras_info]) + 1)
    widths[0] = pixel_gap
    heights[0] = pixel_gap

    for cam in cameras_info.keys():
        width = cameras_info[cam]['Resolution'][0]
        height = cameras_info[cam]['Resolution'][1]
        x_pos = cameras_info[cam]['Position'][0]
        y_pos = cameras_info[cam]['Position'][1]
        if widths[x_pos] < width:
            widths[x_pos] = width + pixel_gap * x_pos
        if heights[y_pos] < height:
            heights[y_pos] = height + pixel_gap * y_pos
    widths -= int(pixel_gap / 2)
    heights -= int(pixel_gap / 2)
    final_resolution = (int(widths.sum()), int(heights.sum()), int(image_depth))

    result_frame = np.zeros(final_resolution)


def initialise(worker_object):
    global vis
    global cameras_setup_string
    global pixel_gap
    global image_depth

    try:
        visualisation_on = worker_object.parameters[0]
        cameras_setup_string = worker_object.parameters[1]
        image_depth = worker_object.parameters[2]
        pixel_gap = worker_object.parameters[3]
    except:
        return False

    vis = VisualisationDPG(_node_name=worker_object.node_name, _node_index=worker_object.node_index,
                           _visualisation_type='Image', _buffer=1)
    vis.visualisation_on = visualisation_on

    setup_output_frame()

    worker_object.savenodestate_create_parameters_df(visualisation_on=vis.visualisation_on,
                                                     cameras_setup_string=cameras_setup_string,
                                                     image_depth=image_depth, pixel_gap=pixel_gap)

    return True


def do_the_concatenation(cam, image_in):
    global cameras_info
    global result_frame
    global widths
    global heights

    image_in = np.ascontiguousarray(image_in) / 255

    pos_x = cameras_info[cam]['Position'][0]
    pos_y = cameras_info[cam]['Position'][1]
    start_x = int(widths[:pos_x].sum())
    end_x = start_x + cameras_info[cam]['Resolution'][0]
    start_y = int(heights[:pos_y].sum())
    end_y = start_y + cameras_info[cam]['Resolution'][1]
    result_frame[start_x:end_x, start_y:end_y, :] = image_in


def concatenate_frames(data, parameters):
    global cameras_info
    global result_frame

    topic = data[0].decode('utf-8')
    data_in = data[1:]
    image_in = Socket.reconstruct_array_from_bytes_message_cv2correction(data_in)
    for cam in cameras_info.keys():
        if cam in topic:
            do_the_concatenation(cam, image_in)
            break

    vis.visualisation_on = worker_object.parameters[0]
    vis.visualise(result_frame)

    return [result_frame]


def on_end_of_life():
    global vis
    vis.end_of_life()


if __name__ == "__main__":
    worker_object = gu.start_the_transform_worker_process(concatenate_frames, on_end_of_life, initialise)
    worker_object.start_ioloop()
