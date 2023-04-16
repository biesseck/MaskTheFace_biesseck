import sys, os
import glob
import numpy as np
import cv2
import argparse

import dlib

# sys.path.append('../')
sys.path.append('.')
from utils.aux_functions import *


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to either the folder containing images or the image itself",
    )

    parser.add_argument(
        "--input_ext",
        type=str,
        default="jpg|png",
        help="",
    )

    parser.add_argument(
        "--output_ext",
        type=str,
        default="png",
        help="",
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default="random",
        # choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],
        choices=["surgical", "N95", "KN95", "cloth", "gas", "random", "all"],
        help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Type of the pattern. Available options in masks/textures",
    )

    parser.add_argument(
        "--pattern_weight",
        type=float,
        default=0.5,
        help="Weight of the pattern. Must be between 0 and 1",
    )

    parser.add_argument(
        "--color",
        type=str,
        default="#0473e2",
        help="Hex color value that need to be overlayed to the mask",
    )

    parser.add_argument(
        "--color_weight",
        type=float,
        default=0.5,
        help="Weight of the color intensity. Must be between 0 and 1",
    )

    parser.add_argument(
        "--code",
        type=str,
        # default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green",
        default="",
        help="Generate specific formats",
    )

    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
    )

    args = parser.parse_args()
    return args


def get_images_paths(path='', exts='jpg|png'):
    img_paths = []
    for ext in exts.split('|'):
        img_paths += glob.glob(path + '/*/*.' + ext)
        # print('img_paths:', img_paths)
        # sys.exit(0)
    img_paths.sort()
    return img_paths


def init_facedetector_and_maskcode(args):
    # Set up dlib face detector and predictor
    # args.detector = dlib.get_frontal_face_detector()
    args.detector = dlib.cnn_face_detection_model_v1('./dlib_models/mmod_human_face_detector.dat')
    
    path_to_dlib_model = "./dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()

    args.predictor = dlib.shape_predictor(path_to_dlib_model)

    '''
    # Extract data from code
    mask_code = "".join(args.code.split()).split(",")
    args.code_count = np.zeros(len(mask_code))
    args.mask_dict_of_dict = {}

    for i, entry in enumerate(mask_code):
        mask_dict = {}
        mask_color = ""
        mask_texture = ""
        mask_type = entry.split("-")[0]
        if len(entry.split("-")) == 2:
            mask_variation = entry.split("-")[1]
            if "#" in mask_variation:
                mask_color = mask_variation
            else:
                mask_texture = mask_variation
        mask_dict["type"] = mask_type
        mask_dict["color"] = mask_color
        mask_dict["texture"] = mask_texture
        args.mask_dict_of_dict[i] = mask_dict
    '''
    return args


def get_arcface_input(aimg):
    input_mean = 127.5
    input_std = 127.5

    # aimg = face_align.norm_crop(img, landmark=face.kps)
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    # return blob[0], aimg
    return blob[0]


def apply_mask_to_face_images(args, img_paths):
    output_path = args.path + '_masked'
    if not os.path.exists(output_path): os.makedirs(output_path)
    for j, image_path in enumerate(img_paths):
        subj_path = '/'.join(image_path.replace(args.path, output_path).split('/')[:-1])
        if not os.path.exists(subj_path): os.makedirs(subj_path)
        print(f'{j}/{len(img_paths)-1} - image_path:', image_path)
        # image_path = args.path                   # original
        # write_path = args.path.rsplit(".")[0]    # original
        # write_path = image_path.rsplit(".")[0]   # Bernardo
        write_path = image_path.replace(args.path, output_path).rsplit(".")[0]   # Bernardo
        if is_image(image_path):
            # Proceed if file is image
            # masked_images, mask, mask_binary_array, original_image
            masked_image, mask, mask_binary_array, original_image = mask_image(
                image_path, args
            )
            for i in range(len(mask)):
                # w_path = write_path + "_" + mask[i] + "." + image_path.rsplit(".")[1]  # original
                w_path = write_path + "_" + mask[i] + "." + args.output_ext              # Bernardo
                img = masked_image[i]
                blob = get_arcface_input(img)

                print('    Savind masked:', w_path, ' - shape:', img.shape)
                cv2.imwrite(w_path, img)

                blob_path = w_path.replace(w_path.rsplit(".")[1], 'npy')
                print('    Savind blob:  ', blob_path, ' - shape:', blob.shape)
                np.save(blob_path, blob)

                print('')
                # sys.exit(0)
        print('-----------------')



if __name__ == '__main__':
    args = get_args()
    print('args:', args)

    # get folders subjects
    img_paths = get_images_paths(args.path, exts=args.input_ext)
    # print('img_paths:', img_paths)
    # sys.exit(0)

    args = init_facedetector_and_maskcode(args)

    apply_mask_to_face_images(args, img_paths)

    # create new MICA image_paths for each dataset