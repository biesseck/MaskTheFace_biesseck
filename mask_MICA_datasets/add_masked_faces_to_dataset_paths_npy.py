import sys, os
import glob
import numpy as np
import argparse


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
        "--path_npy",
        type=str,
        default="/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/datasets/image_paths/LYHM.npy",
        help="",
    )

    args = parser.parse_args()
    return args


def get_dataset_paths_dict(path_npy='LYHM.npy'):
    dataset_paths_dict = np.load(path_npy, allow_pickle=True).item()
    return dataset_paths_dict



def get_images_paths(path='', exts='jpg|png'):
    subjs = []
    subjs = sorted(os.listdir(path))
    # subjs = glob.glob(path + '/*')
    # print('subjs:', subjs)
    # print('len(subjs):', len(subjs))
    # sys.exit(0)

    img_paths_dict = {}
    for subj in subjs:
        img_paths_dict[subj] = []
        for ext in exts.split('|'):
            img_paths_dict[subj] += glob.glob(path + '/' + subj + '/*.' + ext)
        # print(f'img_paths_dict[{subj}]:', img_paths_dict[subj])
        # print('img_paths_dict:', img_paths_dict)
        # sys.exit(0)
    return subjs, img_paths_dict


if __name__ == '__main__':
    args = get_args()
    print('args:', args)

    # get folders subjects
    subjs_masked, img_paths_dict = get_images_paths(args.path, exts=args.input_ext)
    # print('img_paths_dict:', img_paths_dict)
    # print('subjs:', subjs)
    # sys.exit(0)

    # get dataset paths dict (npy)
    dataset_paths_dict = get_dataset_paths_dict(path_npy=args.path_npy)
    subjs_original = sorted(list(dataset_paths_dict.keys()))

    for subj_original in subjs_original:
        relative_img_paths_list = ['../'+'/'.join(img_path.split('/')[-3:]) for img_path in img_paths_dict[subj_original]]
        # print('relative_img_paths_list:', relative_img_paths_list)
        dataset_paths_dict[subj_original][0].extend(relative_img_paths_list)
        # print('\ndataset_paths_dict[subj_original][0]:', dataset_paths_dict[subj_original][0])
        # print('\ndataset_paths_dict[subj_original]:', dataset_paths_dict[subj_original])
        # sys.exit(0)
    # print('\ndataset_paths_dict:', dataset_paths_dict)

    file_name_masked_npy = args.path_npy.split('/')[-1].split('.')[0] + '_augmented_mask.' + args.path_npy.split('/')[-1].split('.')[1]
    path_masked_npy = '/'.join(args.path_npy.split('/')[:-1]) + '/' + file_name_masked_npy
    print('Saving new \'npy\' dataset paths file:', path_masked_npy)
    np.save(path_masked_npy, dataset_paths_dict)
    
    print('Finished!')