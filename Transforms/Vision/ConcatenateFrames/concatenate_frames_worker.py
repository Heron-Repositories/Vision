
import sys
from os import path
import numpy as np

current_dir = path.dirname(path.abspath(__file__))
while path.split(current_dir)[-1] != r'Heron':
    current_dir = path.dirname(current_dir)
sys.path.insert(0, path.dirname(current_dir))

from Heron.communication.socket_for_serialization import Socket
from Heron import general_utils as gu
from Heron.gui.visualisation_dpg import VisualisationDPG

vis: VisualisationDPG
cameras_info: dict
cameras_setup_string: str
image_depth: int
pixel_gap: int
result_frame: np.ndarray
widths: np.ndarray
heights: np.ndarray
result_frame_info: np.ndarray
indices_of_frames_in:np.ndarray


def setup_output_frame():
    global cameras_setup_string
    global cameras_info
    global result_frame
    global pixel_gap
    global image_depth
    global widths
    global heights
    global result_frame_info
    global indices_of_frames_in

    cameras_info = {}
    cameras_setup_list = cameras_setup_string.split(', ')
    for c in cameras_setup_list:
        index = int(c.split(':')[0])
        info = {'Resolution': [int(i) for i in c.split(':')[1].split('x')],
                'Position': [int(i) for i in c.split(':')[2].split('x')]}
        cameras_info[index] = info

    widths = np.zeros(np.max([cameras_info[c]['Position'][1] for c in cameras_info]) + 1)
    heights = np.zeros(np.max([cameras_info[c]['Position'][0] for c in cameras_info]) + 1)
    widths[0] = pixel_gap
    heights[0] = pixel_gap

    for cam in cameras_info.keys():
        width = cameras_info[cam]['Resolution'][0]
        height = cameras_info[cam]['Resolution'][1]
        x_pos = cameras_info[cam]['Position'][1]
        y_pos = cameras_info[cam]['Position'][0]
        if widths[x_pos] < width:
            widths[x_pos] = width + pixel_gap * x_pos
        if heights[y_pos] < height:
            heights[y_pos] = height + pixel_gap * y_pos
    widths -= int(pixel_gap / 2)
    heights -= int(pixel_gap / 2)

    final_resolution = (int(heights.sum()), int(widths.sum()), int(image_depth))

    result_frame = np.zeros(final_resolution).astype(np.uint8)

    indices_of_frames_in = np.zeros(len(cameras_info))
    result_frame_info = np.empty((len(cameras_info), 4))
    for cam in cameras_info.keys():
        pos_x = cameras_info[cam]['Position'][1]
        pos_y = cameras_info[cam]['Position'][0]
        start_x = int(widths[:pos_x].sum())
        end_x = start_x + cameras_info[cam]['Resolution'][0]
        start_y = int(heights[:pos_y].sum())
        end_y = start_y + cameras_info[cam]['Resolution'][1]
        result_frame_info[cam, :] = np.array([start_x, end_x, start_y, end_y])
    result_frame_info = result_frame_info.astype(int)


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


def concatenate_frames(data, parameters):
    global cameras_info
    global result_frame
    global result_frame_info
    global indices_of_frames_in
    
    topic = data[0].decode('utf-8')
    data_in = data[1:]
    image_in = Socket.reconstruct_data_from_bytes_message(data_in)

    for cam in cameras_info.keys():
        if 'Camera##{}'.format(cam) in topic:
            indices_of_frames_in[cam] += 1
            result_frame[result_frame_info[cam, 2]:result_frame_info[cam, 3],
                         result_frame_info[cam, 0]:result_frame_info[cam, 1],
                         :] = image_in
            worker_object.savenodestate_update_substate_df(camera=cam, frame=indices_of_frames_in[cam])
            break

    vis.visualisation_on = worker_object.parameters[0]
    if vis.visualisation_on:
        vis.visualise(result_frame)

    return [result_frame]


def on_end_of_life():
    global vis
    vis.end_of_life()


if __name__ == "__main__":
    worker_object = gu.start_the_transform_worker_process(concatenate_frames, on_end_of_life, initialise)
    worker_object.start_ioloop()
