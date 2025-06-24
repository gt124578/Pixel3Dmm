


# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import argparse
import os
import random
import traceback
from glob import glob
from pathlib import Path
from PIL import Image
from pixel3dmm import env_paths

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from skimage.io import imread
from tqdm import tqdm
#from retinaface.pre_trained_models import get_model
#from retinaface.utils import vis_annotations
#from matplotlib import pyplot as plt


from pixel3dmm.preprocessing.MICA.configs.config import get_cfg_defaults
from pixel3dmm.preprocessing.MICA.datasets.creation.util import get_arcface_input, get_center, draw_on
from pixel3dmm.preprocessing.MICA.utils import util
from pixel3dmm.preprocessing.MICA.utils.landmark_detector import LandmarksDetector, detectors
from pixel3dmm import env_paths


#model = get_model("resnet50_2020-07-20", max_size=512)
#model.eval()


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


def process(args, app, image_size=224, draw_bbox=False):
    dst = Path(args.a)
    dst.mkdir(parents=True, exist_ok=True)
    processes = []
    image_paths = sorted(glob(args.i + '/*.*'))#[:1]
    image_paths = image_paths[::max(1, len(image_paths)//10)]
    for image_path in tqdm(image_paths):
        name = Path(image_path).stem
        img = cv2.imread(image_path)


        # FOR pytorch retinaface use this: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # I had issues with onnxruntime!
        bboxes, kpss = app.detect(img)

        #annotation = model.predict_jsons(img)
        #Image.fromarray(vis_annotations(img, annotation)).show()

        #bboxes = np.stack([np.array( annotation[0]['bbox'] + [annotation[0]['score']] ) for i in range(len(annotation))], axis=0)
        #kpss = np.stack([np.array( annotation[0]['landmarks'] ) for i in range(len(annotation))], axis=0)
        if bboxes.shape[0] == 0:
            logger.error(f'[ERROR] Face not detected for {image_path}')
            continue
        i = get_center(bboxes, img)
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]

        ##for ikp in range(kps.shape[0]):
        #    img[int(kps[ikp][1]), int(kps[ikp][0]), 0] = 255
        #    img[int(kpss_[0][ikp][1]), int(kpss_[0][ikp][0]), 1] = 255
        #Image.fromarray(img).show()
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        blob, aimg = get_arcface_input(face, img)
        file = str(Path(dst, name))
        np.save(file, blob)
        processes.append(file + '.npy')
        cv2.imwrite(file + '.jpg', face_align.norm_crop(img, landmark=face.kps, image_size=image_size))
        if draw_bbox:
            dimg = draw_on(img, [face])
            cv2.imwrite(file + '_bbox.jpg', dimg)

    return processes


def to_batch(path):
    src = path.replace('npy', 'jpg')
    if not os.path.exists(src):
        src = path.replace('npy', 'png')

    image = imread(src)[:, :, :3]
    image = image / 255.
    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
    image = torch.tensor(image).cuda()[None]

    arcface = np.load(path)
    arcface = torch.tensor(arcface).cuda()[None]

    return image, arcface


def load_checkpoint(args, mica):
    checkpoint = torch.load(args.m, weights_only=False)
    if 'arcface' in checkpoint:
        mica.arcface.load_state_dict(checkpoint['arcface'])
    if 'flameModel' in checkpoint:
        mica.flameModel.load_state_dict(checkpoint['flameModel'])


def main(args, mica, app):

    faces = mica.flameModel.generator.faces_tensor.cpu()
    Path(args.o).mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        logger.info(f'Processing has started...')
        paths = process(args, app, draw_bbox=False)
        for path in tqdm(paths):
            name = Path(path).stem
            images, arcface = to_batch(path)
            codedict = mica.encode(images, arcface)
            opdict = mica.decode(codedict)
            meshes = opdict['pred_canonical_shape_vertices']
            code = opdict['pred_shape_code']
            lmk = mica.flameModel.generator.compute_landmarks(meshes)

            mesh = meshes[0]
            landmark_51 = lmk[0, 17:]
            landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

            dst = Path(args.o, name)
            dst.mkdir(parents=True, exist_ok=True)
            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
            np.save(f'{dst}/identity', code[0].cpu().numpy())
            np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
            np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)

        logger.info(f'Processing finished. Results has been saved in {args.o}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    parser.add_argument('-video_name', required=True, type=str)
    parser.add_argument('-a', default='demo/arcface', type=str, help='Processed images for MICA input')
    parser.add_argument('-m', default=f'{env_paths.MICA_TAR_ASSET}', type=str, help='Pretrained model path')

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    args.i = f'{env_paths.PREPROCESSED_DATA}/{args.video_name}/cropped/'
    args.o = f'{env_paths.PREPROCESSED_DATA}/{args.video_name}/mica/'
    if os.path.exists(f'{env_paths.PREPROCESSED_DATA}/{args.video_name}/mica/'):
        if len(os.listdir(f'{env_paths.PREPROCESSED_DATA}/{args.video_name}/mica/')) >= 10:
            print(f'''
                            <<<<<<<< ALREADY COMPLETE MICA PREDICTION FOR {args.video_name}, SKIPPING >>>>>>>>
                            ''')
            exit()

    # instantiate models outside main
    device = 'cuda'
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, device)
    load_checkpoint(args, mica)
    mica.eval()

    app = LandmarksDetector(model=detectors.RETINAFACE)

    main(args, mica, app)
