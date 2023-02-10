
import os
import sys
from os import path

current_dir = path.dirname(path.abspath(__file__))
while path.split(current_dir)[-1] != r'Heron':
    current_dir = path.dirname(current_dir)
sys.path.insert(0, path.dirname(current_dir))

from Heron import general_utils as gu
Exec = os.path.abspath(__file__)


# <editor-fold desc="The following code is called from the GUI process as part of the generation of the node.
# It is meant to create node specific elements (not part of a generic node).
# This is where a new node's individual elements should be defined">

"""
Properties of the generated Node
"""
BaseName = 'Concatenate Frames'
NodeAttributeNames = ['Parameters', 'Frames In', 'Frame Out']
NodeAttributeType = ['Static', 'Input', 'Output']
ParameterNames = ['Visualisation', 'Cam Node Index:Cam Resolution(w x h x c):Cam Position(row x column), ...',
                  'Image Depth', 'Pixel Gap']
ParameterTypes = ['bool', 'str', int, int]
ParametersDefaultValues = [False, '0:1280x720:1x1, 1:1920x1080:1x2', 4, 10]
WorkerDefaultExecutable = os.path.join(os.path.dirname(Exec), 'concatenate_frames_worker.py')
# </editor-fold>


# <editor-fold desc="The following code is called as its own process when the editor starts the graph">
if __name__ == "__main__":
    concatenate_frames_com = gu.start_the_transform_communications_process(NodeAttributeType, NodeAttributeNames)
    gu.register_exit_signals(concatenate_frames_com.on_kill)
    concatenate_frames_com.start_ioloop()

# </editor-fold>
