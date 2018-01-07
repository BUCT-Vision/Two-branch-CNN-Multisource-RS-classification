# -*- coding: utf-8 -*-
import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse 
from data_util import *
from models import *
from ops import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

# trained weight
_lidar_weights = "logs/weights/TRENTO_lidar_weights-0.8517.h5"
_hsi_weights = "logs/weights/TRENTO_hsi_weights-0.9535.h5"
_full_weights = "logs/weights/Hou_weights_finetune-0.8798.h5"
# save weights
_weights_h = "logs/weights/TRENTO_hsi_weights.h5"
_weights_l = "logs/weights/TRENTO_lidar_weights.h5"
_weights = "logs/weights/Salinas_weights_"+str(2*r+1)+".h5"

_TFBooard = 'logs/events/'


parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    type=str,
                    # default='finetune',
                    help='hsi,lidar,finetune')
parser.add_argument('--test',
                    type=str,
                    # default='finetune',
                    help='hsi,lidar,finetune')
parser.add_argument('--modelname', type=str,
                    default='./logs/weights/models.h5', help='final model save name')
parser.add_argument('--epochs',type=int,
                    default=20,help='number of epochs')
args = parser.parse_args()

if not os.path.exists('logs/weights/'):
    os.makedirs('logs/weights/')

if not os.path.exists(_TFBooard):
    # shutil.rmtree(_TFBooard)
    os.mkdir(_TFBooard)

def train_lidar(model):

    # # create train data
    creat_train(validation=False)
    creat_train(validation=True)

    Xl_train = np.load('../file/train_Xl.npy')
    # Xh_train = np.load('../file/train_Xh.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('../file/train_Y.npy'))

    Xl_val = np.load('../file/val_Xl.npy')
    # Xh_val = np.load('../file/val_Xh.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('../file/val_Y.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights_l, verbose=1, save_best_only=True)
    
    # if you need TTensorboard while training phase just uncomment 
    # TFBoard = TensorBoard(
    #     log_dir=_TFBooard, write_graph=True, write_images=False)
    # model.fit([Xl_train], Y_train, batch_size=BATCH_SIZE, class_weight=cls_weights, epochs=args.epochs,
    #           callbacks=[model_ckt, TFBoard], validation_data=([Xl_val], Y_val))
    
    model.fit([Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_data=([Xl_val], Y_val))
    scores = model.evaluate([Xl_val], Y_val,batch_size=100)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)

def train_hsi(model):

    # # create train data
    creat_train(validation=False)
    creat_train(validation=True)

    # Xl_train = np.load('../file/train_Xl.npy')
    Xh_train = np.load('../file/train_Xh.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('../file/train_Y.npy'))

    # Xl_val = np.load('../file/val_Xl.npy')
    Xh_val = np.load('../file/val_Xh.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('../file/val_Y.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights_h, verbose=1, save_best_only=True)
    # if you need tensorboard while training phase just change train fit like 
    # TFBoard = TensorBoard(
    #     log_dir=_TFBooard, write_graph=True, write_images=False)
    # model.fit([Xh_train, Xh_train[:, r, r, :, np.newaxis]], Y_train, batch_size=BATCH_SIZE, class_weight=cls_weights,
    #           epochs=args.epochs, callbacks=[model_ckt, TFBoard], validation_data=([Xh_val, Xh_val[:, r, r, :, np.newaxis]], Y_val))

    model.fit([Xh_train, Xh_train[:, r, r, :, np.newaxis]], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_data=([Xh_val, Xh_val[:, r, r, :,np.newaxis]], Y_val))
    scores = model.evaluate(
        [Xh_val, Xh_val[:, r, r, :, np.newaxis]], Y_val, batch_size=100)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)

def train_full(model):
    # # create train data
    creat_train(validation=False)
    creat_train(validation=True)

    Xl_train = np.load('../file/train_Xl.npy')
    Xh_train = np.load('../file/train_Xh.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('../file/train_Y.npy'))

    Xl_val = np.load('../file/val_Xl.npy')
    Xh_val = np.load('../file/val_Xh.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('../file/val_Y.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights, verbose=1, save_best_only=True)
    # if you need TTensorboard while training phase just uncomment 
    # TFBoard=TensorBoard(log_dir=_TFBooard,write_graph=True,write_images=False)
    # model.fit([Xl_train], Y_train, batch_size=BATCH_SIZE,class_weight=cls_weights, epochs=args.epochs, callbacks=[model_ckt,TFBoard], validation_data=([Xl_val], Y_val))
    
    model.fit([Xh_train,Xh_train[:,r,r,:,np.newaxis],Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_data=([Xh_val,Xh_val[:,r,r,:,np.newaxis],Xl_val], Y_val))
    scores = model.evaluate([Xh_val,Xh_val[:,r,r,:,np.newaxis],Xl_val], Y_val,batch_size=100)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)


def test(network):
    if network =='lidar':
        model = lidar_branch()
        model.load_weights(_weights_l)
        [Xl, Xh] = make_cTest()
        pred = model.predict([Xl])
    if network == 'hsi':
        model = hsi_branch()
        model.load_weights(_weights_h)
        [Xl, Xh] = make_cTest()
        pred = model.predict([Xh,Xh[:,r,r,:,np.newaxis]])
    if network == 'finetune':
        model =finetune_Net()
        model.load_weights(_weights)
        [Xl, Xh] = make_cTest()
        pred = model.predict([Xh,Xh[:,r,r,:,np.newaxis],Xl])
    # Xh,Xh[:,r,r,:,np.newaxis],
    np.save('pred.npy',pred)
    acc,kappa = cvt_map(pred,show=False)
    print('acc: {:.2f}%  Kappa: {:.4f}'.format(acc,kappa))


def main():
    if args.train == 'lidar':
        model = lidar_branch()
        imgname = 'lidar_model.png'
        visual_model(model, imgname)
        train_lidar(model)
    if args.train == 'hsi':
        model = hsi_branch()
        imgname = 'hsi_model.png'
        visual_model(model, imgname)
        train_hsi(model)
    if args.train == 'finetune':
        model = finetune_Net(hsi_weight=None,
                             lidar_weight=None, 
                             trainable=False)
        imgname = 'model.png'
        visual_model(model, imgname)
        train_full(model)
    #test phase
    if args.test == 'lidar':
        start = time.time()
        test('lidar')
        print('elapsed time:{:.2f}s'.format(time.time() - start))
    if args.test == 'hsi':
        start = time.time()
        test('hsi')
        print('elapsed time:{:.2f}s'.format(time.time() - start))
    if args.test == 'finetune':
        start = time.time()
        test('finetune')
        print('elapsed time:{:.2f}s'.format(time.time() - start))

if __name__ == '__main__':
    main()
    
