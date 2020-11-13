from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import numpy as np
import os
import cv2
from yolo_utils import decode_netout, compute_overlap, compute_ap
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from yolo_preprocessing import YoloBatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#from yolo_backend import TinyYoloFeature
import keras
import sys
import matplotlib.pyplot as plt
#import threading

class SpecialYOLO(object):
    def __init__(self, input_width,
                       input_height,
                       labels,
                       max_kpp_per_image ):

        self.input_width = input_width
        self.input_height = input_height

        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.nb_kpp   = 1  #predefined number of keypoint pairs per grid cell
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors = [0.5,0.5]  # vorläufig nur ein kp-Anchor je grid cell
        self.max_kpp_per_image = max_kpp_per_image  #kpp = key point pairs

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image    = Input(shape=(self.input_height, self.input_width, 1))
        #self.true_kpps = Input(shape=(1, 1, 1, max_kpp_per_image , 4))

        # Testaus: vormals: self.feature_extracot ist ein eigenes Model, das nicht trainiert wurde
        #self.feature_extractor = TinyYoloFeature(self.input_size)

        #print("model_1 output shape=", self.feature_extractor.get_output_shape())
        #self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        #features = self.feature_extractor.extract(input_image)
        # Testaus Ende

        # Ersatz für Testaus:
        # make feature extraction layers
        #input_image = Input(shape=(self.input_size, self.input_size, 3))

        num_layer = 0;

        # stack 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_' + str( num_layer ), use_bias=False)(input_image)  #16
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        num_layer += 1

        # stack 2
        for i in range(0,2):  #(0,2)
            #x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)  #32
            #x = BatchNormalization(name='norm_' + str(num_layer))(x)
            #x = LeakyReLU(alpha=0.1)(x)
            #num_layer += 1

            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)  #32
            x = BatchNormalization(name='norm_' + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            num_layer += 1

        # stack 3
        for i in range(0,10):  #vormals (0,1)
            x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)  #32
            x = BatchNormalization(name='norm_' + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            num_layer += 1

        x = Conv2D(3+1+self.nb_class, (3,3), strides=(1,1), padding='same', name='conv_'+str( num_layer ), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        # make the object detection layer
        output = Conv2D(self.nb_kpp * (3 + 1 + self.nb_class),
                        (1,1), strides=(1,1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(x)


        print( "x.shape=", x.shape.as_list() )
        self.grid_h = x.shape.as_list()[1]
        self.grid_w = x.shape.as_list()[2]

        print( "self.grid_h, self.grid_w=", self.grid_h, self.grid_w )

        output = Reshape((self.grid_h, self.grid_w, self.nb_kpp, 3 + 1 + self.nb_class))(x)

        #output = Lambda(lambda args: args[0])([output, self.true_kpps])

        print( "model_1 input shape=", input_image.shape )
        print( "model_2 output shape=", output.shape )

        #self.model = Model([input_image, self.true_kpps], output)
        self.model = Model(inputs=input_image, outputs=output)
        #self.model.load_weights( "transparent.h5" )


        # initialize the weights of the detection layer
        #layer = self.model.layers[-4]  #detection layer
        #weights = layer.get_weights()

        #new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        #new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)

        #layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self.model.summary(positions=[.25, .60, .80, 1.])
        tf.logging.set_verbosity(tf.logging.INFO)  ## testein


    def custom_loss(self, y_true, y_pred): #y_true und y_pred sind die Daten zum gesamten Batch.
        # shape für y_pred (y_true sollte gleiches shape haben): <batch_size> <gridsize_x> <gridsize_y> <nb_anchors> <x0 y0 x1 y1 conf classes one-hot>
        # y_true, y_pred sind die Daten für einen ganzen batch
        # y_true sind Grid-Koordinaten (hier 0...4)
        # y_pred sind cell-Koordinaten (0...1 wenn innerhalb der Cell)

        # netout muss in image_width und image_height-Einheiten sein, d.h. im Intervall [0...1] liegen

        # y_true = tf.Print( y_true, [1], message="***start*** \n", summarize=10000 )

        # y_true = tf.Print( y_true, [y_true], message="y_true= \n", summarize=10000 )
        #y_pred = tf.Print( y_pred, [y_pred], message="y_pred= \n", summarize=10000 )

        mask_shape = tf.shape(y_true)[:4] #mask_shape ist dann (batch_size, nb_grid_x, nb_grid_y, nb_anchors)

        #cell_x und cell_y enthalten danach zu jeder grid_cell ihre jeweiligen x- und y-Koordinaten als lookup-Koordinaten
        cell_x = tf.to_float( tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))  #gleiche dimensionszahl wie y_pred
        cell_y = tf.to_float( tf.reshape( tf.transpose( tf.reshape( tf.tile( tf.range(self.grid_h), [self.grid_w] ),(self.grid_w, self.grid_h))),(1, self.grid_h, self.grid_w, 1, 1)))


        #cell_x = tf.Print( cell_x, [cell_x], message="cell_x=", summarize=10000 )
        #cell_y = tf.Print( cell_y, [cell_y], message="cell_y=", summarize=10000 )

        # cell_grid enthält diese Koordinaten-LUT für jeden batch und für jeden Keypoint
        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_kpp, 1])  #zählt erstes Element zuerst hoch [[0 0][1 0][2 0][3 0]][[0 1][02]...]...
        # cell_grid = tf.Print( cell_grid, [cell_grid], message="cell_grid=", summarize=10000 )
        # gleiches shape wie y_pred ohne classes
        coord_mask = tf.zeros(mask_shape, dtype='float32')
        conf_mask  = tf.zeros(mask_shape, dtype='float32')
        class_mask = tf.zeros(mask_shape, dtype='float32')

        seen = tf.Variable(0., dtype='float32')
        total_recall = tf.Variable(0., dtype='float32')

        """
        Adjust prediction
        """
        ### adjust predicted keypoint pair coordinates

        pred_kp0_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid  # transform to grid coordinates, predicted keypoint0 grid coordinates are the first two elements in last dimension
        #pred_kp1_xy = y_pred[..., 2:4] # predicted keypoint1 grid coordinates are the elements 2+3 in last dimension
        pred_alpha = (y_pred[..., 2:3])  # predicted keypoint1 grid coordinates are the elements 2+3 in last dimension

        ### adjust (=limit to [0...1]) predicted confidence
        pred_kpp_conf = tf.sigmoid(y_pred[..., 3]) #predicted keypoint pair confidence
        # pred_kpp_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust predicted class probabilities
        pred_kpp_class = y_pred[..., 4:]  # one or more classes starting with element 5 in last dimension
        # true_kpp_class = y_true[..., 5:]


        """
        Adjust ground truth
        """
        ### keypoint0 x and y
        true_kp0_xy = y_true[..., 0:2] # unit grid cells, LUC


        ### keypoint1 x and y
        #true_kp1_xy = y_true[..., 2:4] # identity vector
        true_alpha = y_true[..., 2:3] # alpha

        # true_alpha = tf.Print( true_alpha, [true_alpha], message="true_alpha= \n", summarize=10000 )
        # pred_kp1_xy = tf.Print( pred_alpha, [pred_alpha], message="pred_alpha= \n", summarize=10000 )
        # true_kpp_conf = tf.Print( true_kpp_conf, [true_kpp_conf], message="true_kpp_conv= \n", summarize=10000 )

        ### Die argmax-Funktion bestimmt den Index-Tensor der maximalen Argumente über die letzte Dimension. Hier wird aber die letzte Achse angegeben
        ### d.h. vom Shape des Ergebnisvektors wird jeweils das argmax der niedrigsten Dimension bestimmt. Die niedrigste Dimension wird aus dem shape entfernt.
        ### Hier Shape des Ergebnisvektors: (nb_batches, grid_x, grid_y, nb_kpp ), wobei in der letzten Dimension die jeweiligen argmax enthalten sind (hier immer 1)
        true_kpp_class = y_true[..., 4:]
        true_kpp_class_argmax = tf.argmax(y_true[..., 4:], -1)
        #true_kpp_class_argmax = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        # Weiterverwendung ist noch nicht klar: Das ist die Konfidenz zu jedem keypoint-pair, multipliziert mit der coord_scale. Daran wird eine Dimension angehängt.
        coord_mask = tf.expand_dims(y_true[..., 3], axis=-1) * self.coord_scale  #eine Dimension mit Größe 1 am Ende anhängen -> (nb_batches, nb_grid_x, nb_grid_y, nb_anchors, 1, 1)
        #coord_mask = tf.Print( coord_mask, [coord_mask], message="coord_mask=", summarize=10000 )

        #delta_true_kp0_xy = (true_kp0_xy - cell_grid - 0.5)
        #delta_true_kp0_xy = tf.Print( delta_true_kp0_xy, [delta_true_kp0_xy], message="delta_true_kp0_xy= \n", summarize=10000 )

        #delta_true_kp0_xy_pow = tf.pow( delta_true_kp0_xy, 2 )
        #pow_sums = tf.reduce_sum( delta_true_kp0_xy_pow, -1 )

        #pow_sums = tf.Print( pow_sums, [pow_sums], message="pow_sums", summarize = 10000 )

        # gaussian shaped distance function
        #sigma = 2.0
        #eudist_squares = 1.0-tf.exp( -pow_sums)/sigma   #der Ergebnisvektor hat eine Dimension weniger, also (nb_batches, nb_grid_x, nb_grid_y, nb_anchors)

        #eudist_squares = tf.Print( eudist_squares, [eudist_squares], message="eudist_squares= \n", summarize=10000 )

        #eudist_gauss = tf.exp( -eudist_squares/sigma )  # is to be minimized: Gaussian curve over eudist squares (nb_batches, nb_grid_x, nb_grid_y, nb_anchors)

        #eudist_gauss = tf.Print( eudist_gauss, [eudist_gauss], message="eudist_gauss= \n", summarize=10000 )
        ### keypoint-pair confidence

        true_kpp_conf = (y_true[..., 3]) # *(1.0 - eudist_gauss)
        #true_kpp_conf = eudist_gauss #1.0 - eudist_gauss)
        #true_kpp_conf = tf.Print( true_kpp_conf, [true_kpp_conf], message="true_kpp_conf", summarize = 10000 )


        # conf_mask hat am Ende alle Elemente auf no_object_scale oder auf object_scale gesetzt.
        # penalize the confidence difference of all keypoints which are farer away from true keypoints
        conf_mask = conf_mask + 1.0  # alle auf 1 setzen
        #conf_mask = conf_mask + tf.to_float( eudist_gauss < 0.5 )*(1-y_true[...,3])*self.no_object_scale #conf_mask.shape==(nb_batches, nb_grid_x, nb_grid_y, nb_anchors)

        # penalize the confidence difference of all keypoints which are reponsible for corresponding ground truth keypoint0
        conf_mask = conf_mask + y_true[..., 3] * self.object_scale  # die keypoints-enthaltenden cells auf 6 setzen
        #conf_mask = tf.Print( conf_mask, [conf_mask], message="conf_mask= \n", summarize=10000 )

        ### class mask: simply the position of the ground truth boxes (the predictors)
        # nicht klar, ich denke aber dass class_mask einfach den shape (nb_batches, grid_x, grid_y, nb_kpp) hat und aus den Konfidenzen, multipliziert mit class_scale, besteht.
        # der Term " tf.gather(self.class_wt, true_kpp_class_argmax)" liefert meines Erachtens nur lauter 1-Werte, da class_wt nur 1-Werte enthält. Welcher dieser Werte von der
        # Indexmatrix true_kpp_class_argmax gezogen wird, spielt keine Rolle, da immer 1.
        #class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 3] * tf.gather(self.class_wt, true_kpp_class_argmax) * self.class_scale  # übernimmt ones für das argmax#Actually 4
        #class_mask = tf.Print( class_mask, [class_mask], message="class_mask= \n", summarize=10000 )


        """
        Warm-up training
        """
        no_kpp_mask = tf.to_float(coord_mask < self.coord_scale/2.)
        seen = tf.assign_add(seen, 1.)

        true_kp0_xy, true_alpha, coord_mask = tf.cond(tf.less(seen, self.warmup_batches+1),#true_alpha
                              lambda: [true_kp0_xy + (0.5 + cell_grid) * no_kpp_mask,
                                       true_alpha + tf.ones_like(true_alpha) *\
                                       no_kpp_mask,
                                       tf.ones_like(coord_mask)],
                              lambda: [true_kp0_xy,
                                       true_alpha,
                                       coord_mask])


        """
        Finalize the loss
        """



        nb_coord_kpp = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_kpp  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_kpp = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        # loss_kp0_xy    = 1.0 - tf.exp( -tf.reduce_sum(tf.square(true_kp0_xy-pred_kp0_xy)/sigma * coord_mask )) # /(self.nb_kpp + 1e-6) )
        #loss_kp1_xy    = 1.0 - tf.exp( -tf.reduce_sum(tf.square(true_kp1_xy-pred_kp1_xy)/sigma * coord_mask ))# /(self.nb_kpp + 1e-6) )
        #loss_conf  = tf.reduce_sum(tf.square(true_kpp_conf-pred_kpp_conf) * conf_mask)  / (self.nb_kpp  + 1e-6)



        loss_kp0_xy    = tf.reduce_sum(tf.square(true_kp0_xy-pred_kp0_xy) * coord_mask) / (nb_coord_kpp + 1e-6) / 2.
        loss_alpha    = tf.reduce_sum(tf.square(true_alpha-pred_alpha) * coord_mask * self.direction_scale) / (nb_coord_kpp + 1e-6) / 2.
        #loss_alpha   = tf.sigmoid( loss_alpha )
        loss_conf  = tf.reduce_sum(tf.square(true_kpp_conf-pred_kpp_conf) * conf_mask)  / (nb_conf_kpp  + 1e-6) / 2.

        # pred_kpp_class = tf.Print( pred_kpp_class, [pred_kpp_class], message="pred_kpp_class= \n", summarize=10000 )
        # true_kpp_class = tf.Print( true_kpp_class, [true_kpp_class], message="true_kpp_class= \n", summarize=10000 )

        # test
        #class_mask_expanded = tf.expand_dims( class_mask, -1)
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_kpp_class_argmax, logits=pred_kpp_class)
        #class_mask_expanded = tf.expand_dims( class_mask, -1)
        #class_mask_expanded = tf.Print( class_mask_expanded, [class_mask_expanded], message="class_mask_expanded", summarize=10000 )
        #loss_class  = tf.reduce_sum(tf.square(true_kpp_class-pred_kpp_class)*class_mask_expanded) / (nb_class_kpp + 1e-6)#/2. # * tf.expand_dims( class_mask, axis=-1 ))  / (nb_class_kpp  + 1e-6) / 2.


        #loss_class = tf.cond( tf.less( 1, self.nb_class ),
        #                       lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_kpp_class_argmax, logits=pred_kpp_class),
        #                      lambda: tf.squeeze( tf.square( true_kpp_class - pred_kpp_class), 4 ))


        # loss_conf = tf.Print( loss_conf, [loss_conf], message="loss_conf= \n", summarize=10000 )
        # testaus
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_kpp + 1e-6)
        #loss_class = tf.sigmoid( loss_class )
        #loss_class = tf.Print( loss_class, [loss_class], message="loss_class=\n" )


        loss = tf.cond(tf.less(seen, self.warmup_batches+1),
                      lambda: loss_kp0_xy + loss_alpha + loss_conf +  loss_class + 10,
                      lambda: loss_kp0_xy + loss_alpha + loss_conf + loss_class)



        if self.debug:
            nb_true_kpp = tf.reduce_sum(y_true[..., 3])
            nb_pred_kpp = tf.reduce_sum(tf.to_float(true_kpp_conf > 0.5) * tf.to_float(pred_kpp_conf > 0.3))

            current_recall = nb_pred_kpp/(nb_true_kpp + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_kp0_xy], message='Loss Keyp0 \t', summarize=1000)
            loss = tf.Print(loss, [loss_alpha], message='Loss alpha \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)

        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)
        self.model.save(weight_path+"full")

        print( "input layer name=" )
        print( [node.op.name for node in self.model.inputs] )
        print( "output layer name=" )
        print( [node.op.name for node in self.model.outputs] )


        return self.model.output.shape[1:3]

    def normalize(self, image):
        return image / 255.

    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    train_times,    # the number of time to repeat the training set, often used for small datasets
                    valid_times,    # the number of times to repeat the validation set, often used for small datasets
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    direction_scale,
                    saved_weights_name='transparent.h5',
                    debug=False):

        self.batch_size = batch_size

        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale
        self.direction_scale = direction_scale

        self.debug = debug

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_height,
            'IMAGE_W'         : self.input_width,
            'GRID_H'          : self.grid_h,
            'GRID_W'          : self.grid_w,
            'KPP'             : self.nb_kpp,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_KPP_BUFFER' : self.max_kpp_per_image,
        }

        train_generator = YoloBatchGenerator(train_imgs,
                                     generator_config,
                                     norm=self.normalize)
        valid_generator = YoloBatchGenerator(valid_imgs,
                                     generator_config,
                                     norm=self.normalize,
                                     jitter=False)

        self.warmup_batches  = warmup_epochs * (train_times*len(train_generator) + valid_times*len(valid_generator))

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=500000,#2or3
                           mode='min',
                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only = False,
                                     mode='min',
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                                  histogram_freq=0,
                                  #write_batch_performance=True,
                                  write_graph=True,
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################


        self.model.fit_generator(generator        = train_generator,
                                 steps_per_epoch  = len(train_generator) * train_times,
                                 epochs           = warmup_epochs + nb_epochs,
                                 verbose          = 2 if debug else 1,
                                 validation_data  = valid_generator,
                                 validation_steps = len(valid_generator) * valid_times,
                                 callbacks        = [early_stop, checkpoint, tensorboard],
                                 workers          = 3,  # vormals 3
                                 max_queue_size   = 8)#8
                                 #use_multiprocessing = False)

        ############################################
        # Compute mAP on the validation set
        ############################################

        ##### test prediction ###########################
        print( "test prediction start\n" )
        image = cv2.imread("C:\\Users\\Sai\\yolo\\Shaft_Program\\v\\Img_37.bmp")
        image = image[:,:,0] # red channel only
        image = np.expand_dims( image, -1 )
        image = self.normalize( image )

        # x_batch, y_batch = train_generator.__getitem__( 0 )
        # for i in range( 0, x_batch.shape[0] ):
        #     image = x_batch[i]
        #     cv2.imwrite( "data\\aug_images\\augimg_"+str( i ) + ".bmp", image*255 )

        # image = x_batch[0]
        self.predict(image)
        print( "test prediction end\n" )
        ##### test prediction ende ######################
        # average_precisions = self.evaluate(valid_generator)

        # print evaluation
        # for label, average_precision in average_precisions.items():
        #     print(self.labels[label], '{:.4f}'.format(average_precision))
        # print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
        #for i in range( 512):
        #    cv2.imwrite( "data\\aug_images\\augimg_"+str( i ) + ".bmp", train_generator.proto_images[i] )

    def evaluate(self,
                 generator,
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """

        print( "***evaluation***\n" )
        # gather all detections and annotations
        all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self.predict(raw_image)


            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.x0*raw_width, box.y0*raw_height, box.x1*raw_width, box.y1*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

    def predict(self, image):
        print("image.shape=",image.shape)
        image_h, image_w, _  = image.shape
        #image = cv2.resize(image, (self.input_width, self.input_height))
        image = self.normalize(image)

        input_image = image[:,:,::-1] #flip rgb to bgr or vice versa

        input_image = np.expand_dims(input_image, 0)
        #dummy_array = np.zeros((1,1,1,1,self.max_kpp_per_image,4))

        netout = self.model.predict([input_image])[0]# add dummy_array

        #print( "netout=", [netout] )  # print the netout


        netout_decoded = decode_netout(netout, self.anchors, image_w, image_h, self.nb_class)


        for kpp in netout_decoded:
            print( kpp.x0, kpp.y0, kpp.alpha_norm, kpp.c , kpp.classes )

        return netout_decoded
