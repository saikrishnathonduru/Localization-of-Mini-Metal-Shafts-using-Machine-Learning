# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 12:44:59 2018

@author: wgr
"""

import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import imgaug as ia
from keras.utils import Sequence
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from yolo_utils import KeyPointPair, draw_kpp
import math

def read_annotations(img_dir):
    all_imgs = []
    seen_labels = {}

    for ann_file_name in sorted(os.listdir(img_dir)):
        ext = os.path.splitext( ann_file_name )[1]

        if ext == '.txt':
            img = {'object':[]}

            img_file_name = img_dir + ann_file_name  # das ist aber das annotation file
            print("img_file_name=", img_file_name)
            print( "ann_file_name=", ann_file_name )

            # read in predefined order
            file = open( img_file_name, "r" )

            for line in file:  # jede Zeile ist ein Keypoint-Pair zu diesem Bild
                vWords = line.split()
                obj = {}

                img['filename'] = os.path.splitext( img_file_name )[0] + ".bmp"  # image file name
                obj['name'] = vWords[0]  # class name

                if obj['name'] in seen_labels:
                    seen_labels[obj['name']] += 1
                else:
                    seen_labels[obj['name']] = 1

                ## two keypoints which give pose (x0, y0) and direction (x1, y1) in pixel coordinates, all following coordinates are from object polygon and thus ignored
                obj['x0'] = float( vWords[1] )
                obj['y0'] = float( vWords[2] )
                obj['x1'] = float( vWords[3] )
                obj['y1'] = float( vWords[4] )

                img['object'] += [obj]  # zu diesem img das keypoint-pair

            file.close()

            if len(img['object']) > 0:
                all_imgs += [img]
    return all_imgs, seen_labels

class YoloBatchGenerator(Sequence):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       jitter=True,
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm
        self.image_counter = 0

        ia.seed( 1 )


        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.8, aug)

        # hier wird die Augmentation definiert, aber nicht ausgeführt.
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                sometimes( iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode = "edge"

                )),
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        #    iaa.CoarseSalt(0.01, size_percent=(0.002, 0.01)),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-15, 15), per_channel=0), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ])

            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)


    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['x0'], obj['y0'], obj['x1'], obj['y1'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):  # liefert einen kompletten batch(Get a complete batch)
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']


        best_anchor = 0
        #x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1))   # Dies sind alle images im batch hier evtl. channels=3 durch 1 ersetzen

        # TRUE_KPP_BUFFER == max_kpp_per_image (hier z.B. 10 in config), list of keypoints x0, y0, x1, y1 ein Keypointpaar per grid cell, same as in y_batch
        # but in ascending order, box-by-box

        # <batchsize> <BOX == nb_box == len( anchors )//2>, <for each box 4*keypoint pairs +  1*confidency + one-hot labels> == desired network output tensor
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['KPP'], 3+1+len(self.config['LABELS'])))

        i_img = 0
        keypoints_on_images = [] # enthält alle keypoints im Bild, fertig für die Augmentation(contains all keypoints in the image, ready for augmentation)
        images_batch = []

        #keypoints-list aufbauen
        num_images = len( self.images )
        instance_src_index = l_bound % num_images
        for instance_count in range( r_bound - l_bound ):
            train_instance = self.images[instance_src_index]
            # augment input image and fix object's position and size
            # img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            image_name = train_instance['filename']
            img = cv2.imread(image_name)
            img = img[:,:,1]  # green channel only
            img = np.expand_dims( img, -1 )  # wieder eine Dimension anhängen

            images_batch.append( img )

            # construct output from object's x, y, w, h
            true_kpp_index = 0

            keypoints_on_image = []
            all_objs = train_instance['object']
            for obj in all_objs:
                keypoints_on_image.append( ia.Keypoint( x=float(obj['x0']), y=float(obj['y0']) ))
                keypoints_on_image.append( ia.Keypoint( x=float(obj['x1']), y=float(obj['y1']) ))

            keypoints_on_images.append(ia.KeypointsOnImage(keypoints_on_image, shape=img.shape ))
            instance_src_index = (instance_src_index + 1) % num_images

        if self.jitter:
            ia.seed( 134 )
            aug_pipe_det = self.aug_pipe.to_deterministic() # damit die augmentation der images und der keypoints dieselben transformationen bewirken(so that the augmentation of the images and the keypoints effect the same transformations)
            x_batch = aug_pipe_det.augment_images(images_batch) # augmented images
            keypoints_batch_aug = aug_pipe_det.augment_keypoints( keypoints_on_images )  # augmented keypoints
        else:
            x_batch = images_batch
            keypoints_batch_aug = keypoints_on_images

        x_batch = np.reshape( x_batch, (r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1 ) )
        #x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1))
        x_batch = self.norm( x_batch )

        # augmentierte Keypoints in y_batch eintragen
        num_images = len( self.images )
        instance_src_index = l_bound % num_images
        for instance_count in range( r_bound - l_bound ):
            train_instance = self.images[instance_src_index]
            all_objs = train_instance['object']
            obj_count = 0
            for obj in all_objs:
                if obj['name'] in self.config['LABELS']:
                    kp0_x = keypoints_batch_aug[instance_count].keypoints[obj_count*2].x
                    kp0_x = kp0_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    kp0_y = keypoints_batch_aug[instance_count].keypoints[obj_count*2].y
                    kp0_y = kp0_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    kp1_x = keypoints_batch_aug[instance_count].keypoints[obj_count*2+1].x
                    kp1_x = kp1_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    kp1_y = keypoints_batch_aug[instance_count].keypoints[obj_count*2+1].y
                    kp1_y = kp1_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    dx = kp1_x - kp0_x
                    dy = kp1_y - kp0_y

                    alpha = math.atan2( dy, dx )
                    if alpha < 0.0:
                        alpha = alpha + math.pi

                    alpha = alpha/math.pi*0.8 + 0.1  # match to 0.1...0.9

                    # (Determine the grid cell to which the keypoint belongs.)Grid-Cell bestimmen, zu der der Keypoint gehört
                    grid_x = int(np.floor(kp0_x))  #das sind die Grid-Koordinaten, z.B. im 4x4-Grid, in das das Image unterteilt wird
                    grid_y = int(np.floor(kp0_y))



                    if grid_x >= 0 and grid_y >= 0 and grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])   #label-class-number

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        best_anchor = 0  #vorerst nur ein anchor-keypoint je grid_cell
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0] = kp0_x  #keypoint0 in grid-Koordinaten LUC
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 1] = kp0_y
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 2] = alpha
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 3] = 1.  #confidence
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4+obj_indx] = 1.  #one-hot class

                        true_kpp_index += 1
                        true_kpp_index = true_kpp_index % self.config['TRUE_KPP_BUFFER']   #avoid overflow

                        #kp_batch[instance_count,0] = kp0_x
                        #kp_batch[instance_count,1] = kp0_y
                        #kp_batch[instance_count,2] = kp1_x
                        #kp_batch[instance_count,3] = kp1_y

                        # Testausdruck des augmented images mit den beiden Keypoints
                        #img2 = copy.deepcopy( x_batch[instance_count] )
                        #if self.image_counter < 512:
                            #self.proto_images[self.image_counter] = x_batch[instance_count]*255.0
                        #cv2.circle(img2, (int( kp_batch[instance_count,0]/self.config['GRID_W']*self.config['IMAGE_W']), int(kp_batch[instance_count,1]/self.config['GRID_H']*self.config['IMAGE_H'])), 7, (0,255,0), 1)
                        #cv2.circle(img2, (int( kp_batch[instance_count,2]/self.config['GRID_W']*self.config['IMAGE_W']), int(kp_batch[instance_count,3]/self.config['GRID_H']*self.config['IMAGE_H'])), 3, (255,255,0), 1)
                        #cv2.imwrite( "data\\aug_images\\img_kpp_"+str(self.image_counter)+".bmp", img2*255)  #Bild ist hier nicht normalisiert
                        #Testausdruck Ende
                        self.image_counter += 1

                    obj_count += 1
            instance_src_index = (instance_src_index + 1) % num_images


        #print(' new batch created', idx)

        #print( "x_batch", x_batch.shape, " y_batch.shape=", y_batch.shape )
        #for i in range( 0, instance_count ):
        #    img3 = x_batch[i]
        #    cv2.imwrite( "data\\aug_images\\augimg_"+str( self.image_counter ) + ".bmp", img3*255 )
        #    self.image_counter += 1

        return x_batch, y_batch   #image normalized and y_batch in grid coordinates

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)  #shuffle along the first axis only

    def aug_image(self, train_instance, jitter):  # (not executed)wird nicht ausgeführt
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        image = image[:,:,0] #red channel only

        if image is None: print('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])  #deepcopy erstellt eine wirkliche Kopie des gesamten arrays, Koordinaten sind Pixel-Koordinaten

        if jitter:
            ia.seed( 1 )
            # temporary array of all keypoints in this image
            keypoint_arr = []

            # keypoints must be given in pixel coordinates
            for obj in all_objs:
                keypoint_arr.append( ia.Keypoint( x=float(obj['x0'])/self.config['GRID_W']*self.config['IMAGE_W'], y=float(obj['y0'])/self.config['GRID_H']*self.config['IMAGE_H'] ))
                keypoint_arr.append( ia.Keypoint( x=float(obj['x1'])/self.config['GRID_W']*self.config['IMAGE_W'], y=float(obj['y1'])/self.config['GRID_H']*self.config['IMAGE_H'] ))
                keypoint_arr.append( ia.Keypoint( x=float(obj['x0']), y=float(obj['y0']) ))
                keypoint_arr.append( ia.Keypoint( x=float(obj['x1']), y=float(obj['y1']) ))

            keypoints = ia.KeypointsOnImage( keypoint_arr, shape = image.shape )

            image_aug = self.aug_pipe_det.augment_image(image) # augmented image
            keypoints_aug = self.aug_pipe_det.augment_keypoints( [keypoints] )[0]  # augmented keypoints

            image = image_aug

            # push back augmented coordinates
            ikp = 0
            for obj in all_objs:
                obj['x0'] = float( keypoints_aug.keypoints[ikp*2].x ) #/self.config['IMAGE_W']*self.config['GRID_W']
                obj['y0'] = float( keypoints_aug.keypoints[ikp*2].y ) # /self.config['IMAGE_H']*self.config['GRID_H']
                obj['x1'] = float( keypoints_aug.keypoints[ikp*2+1].x ) # /self.config['IMAGE_W']*self.config['GRID_W']
                obj['y1'] = float( keypoints_aug.keypoints[ikp*2+1].y ) # /self.config['IMAGE_H']*self.config['GRID_H']
                ikp += 1

        # resize the image to standard size, Achtung: Hier sind die Keypoints nicht resized!!
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))   # hier wird das verschobene und vorher auf die konfigurierte image_size skalierte Image wieder resized
        # hier einfügen: resize_keypoints, um sie an as resized image anzupassen
        image = self.norm( image )  #Bild normieren
        # image = image[:,:,::-1]  #flip rgb to bgr or vice versa

        return image, all_objs
