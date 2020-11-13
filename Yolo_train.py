import argparse
import os
import numpy as np
from yolo_preprocessing import read_annotations
from yolo_frontend import SpecialYOLO
import json
import sys

# run with command line -c yolo_config.json
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  #"0" for gpu usage

#sys.path.append('C:/Users/thond/AppData/Local/Programs/Python/Python36/Lib/site-packages')

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def _main_(args):
    config_path = args.conf
    print( config_path )

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = read_annotations(config['train']['train_image_folder'] )
    valid_imgs, valid_labels = read_annotations(config['valid']['valid_image_folder'] )

    print( "train_imgs= ", train_imgs )

    # parse annotations of the validation set, if any, otherwise split the training set
    #train_valid_split = int(0.8*len(train_imgs))
    np.random.shuffle(train_imgs)
    np.random.shuffle(valid_imgs)

    #valid_imgs = train_imgs[train_valid_split:]
    #train_imgs = train_imgs[:train_valid_split]

    # check if all labels are contained in train annotations
    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()

    ###############################
    #   Construct the model
    ###############################


    yolo = SpecialYOLO( input_width         = config['model']['input_width'],
                        input_height        = config['model']['input_height'],
                        labels              = config['model']['labels'],
                        max_kpp_per_image   = config['model']['max_kpp_per_image'] )

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    #if os.path.exists(config['train']['pretrained_weights']):
        #print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        #yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process
    ###############################

    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epochs          = config['train']['nb_epochs'],
               learning_rate      = config['train']['learning_rate'],
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = config['train']['warmup_epochs'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               direction_scale    = config['train']['direction_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = config['train']['debug'])

if __name__ == '__main__':
    args = argparser.parse_args()
    #testein
    args.conf = "yolo_config.json"

    _main_(args)
