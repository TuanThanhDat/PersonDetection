
import cv2
import torch
import glob
import numpy as np
import os
from mivolo.predictor import Predictor

class PersonDetector:
    def __init__(self, verbose=True, device='cpu', weight_folder=None):
        self.predictor = Predictor(verbose=False, draw=False, device=device, weight_folder=weight_folder)
    
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.verbose = verbose

    def __get_genders(self, detected_objs):
        return list(filter(lambda x: x is not None, detected_objs.genders))

    def __count_female_male(self, genders):
        count_male = 0
        count_female = 0
        for i in genders:
            if i == 'male':
                count_male += 1
            elif i == 'female':
                count_female += 1
        return (count_female, count_male)

    def __get_n_persons(self, detected_objs):
        return detected_objs.n_persons

    def __detect_image(self, img):
        detected_objs, _ = self.predictor.recognize(img)
        genders = self.__get_genders(detected_objs)
        n_female, n_male = self.__count_female_male(genders)
        numbers = self.__get_n_persons(detected_objs)
        # không lưu giá trị nếu tổng số người lớn hơn 5
        if numbers > 5:
            numbers = -1
            n_female = -1
            n_male = -1
        return (numbers,n_female,n_male)

    def read_img(self, img_path):
        return cv2.imread(img_path)

    def detect_images(self, img_paths):
        result = []
        for path in img_paths:
            img = self.read_img(path)
            result.append(list(self.__detect_image(img)))
        return result

    def __get_image_paths(self, folder_path):
        return sorted(glob.glob(f'{folder_path}/*.jpg'))

    def detect_kf_folders(self, group_path, folder_names, save_path):
        for name in folder_names:
            path = f'{group_path}/{name}'
            img_paths = self.__get_image_paths(path)
            if self.verbose:
                print(f'--Processing video keyframe {path}')
            result = self.detect_images(img_paths)
            save_file = f'{save_path}/{name}.npy'
            np.save(save_file, result)

    def __get_kf_folder_names(self, group_path):
        all_paths = sorted(glob.glob(f'{group_path}/L*'))
        return [os.path.split(path)[-1] for path in all_paths if os.path.isdir(path)]

    def detect_groups(self, db_path, group_names, save_path):
        for name in group_names:
            group_path = f"{db_path}/KeyFrames_{name}"
            folder_names = self.__get_kf_folder_names(group_path)
            if self.verbose:
                print(f'--Processing group {group_path}')
            self.detect_kf_folders(group_path, folder_names, save_path)

    def __get_group_names(self, batch_path):
        all_paths = sorted(glob.glob(f'{batch_path}/KeyFrames_*'))
        return [os.path.split(path)[-1][-3:] for path in all_paths if os.path.isdir(path)]

    def detect_batches(self, db_path, batch_names, save_path):
        # db -> batch -> group -> video -> frame
        for name in batch_names:
            # ví dụ: name = 'batch1'
            batch_path = f'{db_path}/keyframes_{name}'
            group_names = self.__get_group_names(batch_path)
            if self.verbose:
                print(f'Processing {name}:')
            self.detect_groups(batch_path, group_names, save_path)
            if self.verbose:
                print(f'Done processing {name}!!!')