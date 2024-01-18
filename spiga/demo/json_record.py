# import os
# import yaml
# import json
# import cv2
# import numpy as np
# import time
# import pkg_resources
# from datetime import date
# from collections import namedtuple

# # My libs
# import spiga.demo.analyze.track.get_tracker as tr
# import spiga.demo.analyze.extract.spiga_processor as pr_spiga
# from spiga.demo.analyze.analyzer import VideoAnalyzer
# from spiga.demo.visualize.viewer import Viewer

# model_config = 'spiga'
# print_bool = True
# results_path = '/home/lucas/lucas/multimodal_perception_test/'

# # Paths
# video_out_path_dft = pkg_resources.resource_filename('spiga', 'demo/outputs')
# if not os.path.exists(video_out_path_dft):
#     os.makedirs(video_out_path_dft)

# def main():
#     import argparse
#     pars = argparse.ArgumentParser(description='Face App')
#     pars.add_argument('-i', '--input', type=str, default='0', help='Video input')
#     pars.add_argument('-d', '--dataset', type=str, default='wflw',
#                       choices=['wflw', '300wpublic', '300wprivate', 'merlrav'],
#                       help='SPIGA pretrained weights per dataset')
#     pars.add_argument('-t', '--tracker', type=str, default='RetinaSort',
#                       choices=['RetinaSort', 'RetinaSort_Res50'], help='Tracker name')
#     pars.add_argument('-sh', '--show',  nargs='+', type=str, default=['fps', 'face_id', 'landmarks', 'headpose'],
#                       choices=['fps', 'bbox', 'face_id', 'landmarks', 'headpose'],
#                       help='Select the attributes of the face to be displayed ')
#     pars.add_argument('-s', '--save', action='store_true', help='Save record')
#     pars.add_argument('-nv', '--noview', action='store_false', help='Do not visualize the window')
#     pars.add_argument('--outpath', type=str, default=video_out_path_dft, help='Video output directory')
#     pars.add_argument('--fps', type=int, default=30, help='Frames per second')
#     pars.add_argument('--shape', nargs='+', type=int, help='Visualizer shape (W,H)')
#     pars.add_argument('--feature', type=str, default='test', help='Feature to be extracted' )
#     args = pars.parse_args()

#     if args.shape:
#         if len(args.shape) != 2:
#             raise ValueError('--shape requires two values: width and height. Ej: --shape 256 256')
#         else:
#             video_shape = tuple(args.shape)
#     else:
#         video_shape = None

#     if not args.noview and not args.save:
#         raise ValueError('No results will be saved neither shown')

#     video_app(args.input, args.feature, spiga_dataset=args.dataset, tracker=args.tracker, fps=args.fps,
#               save=args.save, output_path=args.outpath, video_shape=video_shape, visualize=args.noview, plot=args.show)


# def video_app(input_name, feature, spiga_dataset=None, tracker=None, fps=30, save=False,
#               output_path=video_out_path_dft, video_shape=None, visualize=True, plot=()):

#     # Load video
#     try:
#         capture = cv2.VideoCapture(int(input_name))
#         video_name = None
#         if not visualize:
#             print('WARNING: Webcam must be visualized in order to close the app')
#         visualize = True

#     except:
#         try:
#             capture = cv2.VideoCapture(input_name)
#             video_name = input_name.split('/')[-1][:-4]
#         except:
#             raise ValueError('Input video path %s not valid' % input_name)

#     if capture is not None:
#         # Initialize viewer
#         if video_shape is not None:
#             vid_w, vid_h = video_shape
#         else:
#             vid_w, vid_h = capture.get(3), capture.get(4)
#         viewer = Viewer('face_app', width=vid_w, height=vid_h, fps=fps)
#         if save:
#             viewer.record_video(output_path, video_name)

#         # Initialize face tracker
#         faces_tracker = tr.get_tracker(tracker)
#         faces_tracker.detector.set_input_shape(capture.get(4), capture.get(3))
#         # Initialize processors
#         processor = pr_spiga.SPIGAProcessor(dataset=spiga_dataset)
#         # Initialize Analyzer
#         faces_analyzer = VideoAnalyzer(faces_tracker, processor=processor)
#         userFacialLandmarks = {}
#         # Convert FPS to the amount of milliseconds that each frame will be displayed
#         if visualize:
#             viewer.start_view()
#         while capture.isOpened():
#             ret, frame = capture.read()
#             if ret:
#                 # Process frame
#                 faces_analyzer.process_frame(frame)
        
#                 for i, faceId in enumerate(faces_analyzer.get_attributes('face_id')):
#                     landmarks = np.array(faces_analyzer.get_attributes('landmarks')[i])
#                     userFacialLandmarks[time.time()] = {faceId: landmarks}

    
#                 key = viewer.process_image(frame, drawers=[faces_analyzer], show_attributes=plot)
#                 if key:
#                     break
#             else:
#                 break

#         with open(f"{results_path}{feature}/{date.today()}.json", 'w') as f:
#             json.dump(userFacialLandmarks, f)

#         capture.release()
#         viewer.close()


# if __name__ == '__main__':
#     main()

import os
import json
import cv2
import numpy as np
import time
from datetime import datetime

# My libs
import spiga.demo.analyze.track.get_tracker as tr
import spiga.demo.analyze.extract.spiga_processor as pr_spiga
from spiga.demo.analyze.analyzer import VideoAnalyzer
from spiga.demo.visualize.viewer import Viewer

model_config = 'spiga'
results_path = '/home/lucas/lucas/multimodal_perception_test/json_results/'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():
    import argparse
    pars = argparse.ArgumentParser(description='Face App')
    pars.add_argument('-i', '--input', type=str, default='0', help='Video input')
    pars.add_argument('-d', '--dataset', type=str, default='wflw',
                      choices=['wflw', '300wpublic', '300wprivate', 'merlrav'],
                      help='SPIGA pretrained weights per dataset')
    pars.add_argument('-t', '--tracker', type=str, default='RetinaSort',
                      choices=['RetinaSort', 'RetinaSort_Res50'], help='Tracker name')
    pars.add_argument('-sh', '--show',  nargs='+', type=str, default=['fps', 'face_id', 'landmarks', 'headpose'],
                      choices=['fps', 'bbox', 'face_id', 'landmarks', 'headpose'],
                      help='Select the attributes of the face to be displayed ')
    pars.add_argument('-nv', '--noview', action='store_false', help='Do not visualize the window')
    pars.add_argument('--fps', type=int, default=30, help='Frames per second')
    pars.add_argument('--shape', nargs='+', type=int, help='Visualizer shape (W,H)')
    pars.add_argument('--feature', type=str, default='test', help='Feature to be extracted' )
    args = pars.parse_args()

    if args.shape:
        if len(args.shape) != 2:
            raise ValueError('--shape requires two values: width and height. Ej: --shape 256 256')
        else:
            video_shape = tuple(args.shape)
    else:
        video_shape = None

    video_app(args.input, args.feature, spiga_dataset=args.dataset, tracker=args.tracker, fps=args.fps,
                video_shape=video_shape, visualize=args.noview, plot=args.show)


def video_app(input_name, feature, spiga_dataset=None, tracker=None, fps=30, video_shape=None, visualize=True, plot=()):

    # Load video
    try:
        capture = cv2.VideoCapture(int(input_name))
    except:
        try:
            capture = cv2.VideoCapture(input_name)
        except:
            raise ValueError('Input video path %s not valid' % input_name)

    if capture is not None:
        # Initialize viewer
        if video_shape is not None:
            vid_w, vid_h = video_shape
        else:
            vid_w, vid_h = capture.get(3), capture.get(4)
        viewer = Viewer('face_app', width=vid_w, height=vid_h, fps=fps)
        # Initialize face tracker
        faces_tracker = tr.get_tracker(tracker)
        faces_tracker.detector.set_input_shape(capture.get(4), capture.get(3))
        # Initialize processors
        processor = pr_spiga.SPIGAProcessor(dataset=spiga_dataset)
        # Initialize Analyzer
        faces_analyzer = VideoAnalyzer(faces_tracker, processor=processor)
        userFacialLandmarks = {}
        if visualize:
            viewer.start_view()
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                # Process frame
                faces_analyzer.process_frame(frame)
        
                for i, faceId in enumerate(faces_analyzer.get_attributes('face_id')):
                    landmarks = np.array(faces_analyzer.get_attributes('landmarks')[i])
                    userFacialLandmarks[time.time()] = {faceId: landmarks.tolist()}
                key = viewer.process_image(frame, drawers=[faces_analyzer], show_attributes=plot)
                if key:
                    break

            else:
                break
            print("saved")
            # if the folder does not exist, create it
            if not os.path.exists(f"{results_path}{feature}"):
                os.makedirs(f"{results_path}{feature}")
            with open(f"{results_path}{feature}/{timestamp}.json", 'w') as f:
                json.dump(userFacialLandmarks, f)

        capture.release()
        viewer.close()


if __name__ == '__main__':
    main()