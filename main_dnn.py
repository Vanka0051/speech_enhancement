"""
Summary:  Train, inference and evaluate speech enhancement. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import numpy as np
import os
import pickle
import cPickle
import h5py
import argparse
import time
import glob
#import matplotlib.pyplot as #plt
import tensorflow as tf
import prepare_data as pp_data
import config as cfg
from data_generator import DataGenerator
from data_generator import DataGenerator_h5py
from spectrogram_to_wave import recover_wav
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import load_model


def eval(model, gen, x, y):
    """Validation function. 
    
    Args:
      model: keras model. 
      gen: object, data generator. 
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []
    
    # Inference in mini batch. 
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)
        
    # Concatenate mini batch prediction. 
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # Compute loss. 
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss
    




def eval_h5py(model, gen, path_list):
    """Validation function. 
    
    Args:
      model: keras model. 
      gen: object, data generator. 
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []
    # Inference in mini batch. 
    for (batch_x, batch_y) in gen.generate(path_list):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)
    # Concatenate mini batch prediction. 
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    # Compute loss. 
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss






def train(args):

    """Train the neural network. Write out model every several iterations. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      lr: float, learning rate. 
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    lr = args.lr
    data_type = "IRM"
    # Load data. 
    t1 = time.time()
        #    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "data.h5")
    if data_type=="DM":
        tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "data.h5")
        te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "mixdb" , "data.h5")
    else:
        tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mask_mixdb", "data.h5")
        te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "mask_mixdb" , "data.h5")
    (tr_x, tr_y) = pp_data.load_hdf5(tr_hdf5_path)
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)
    print("Load data time: %s s" % (time.time() - t1,))

    batch_size = 2048
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))

    # Scale data. 
    if True:
        t1 = time.time()
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x = pp_data.scale_on_3d(tr_x, scaler)
        te_x = pp_data.scale_on_3d(te_x, scaler)
        if data_type=="DM":
            tr_y = pp_data.scale_on_2d(tr_y, scaler)
            te_y = pp_data.scale_on_2d(te_y, scaler)
        print("Scale data time: %s s" % (time.time() - t1,))
        
    # Debug plot. 
    if False:
        #plt.matshow(tr_x[0 : 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        #plt.show()
        pause
        
    # Build model
    (_, n_concat, n_freq) = tr_x.shape
    n_hid = 2048

    model = Sequential()
    model.add(Flatten(input_shape=(n_concat, n_freq)))
    model.add(Dense(n_hid, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_freq, activation='linear'))
    model.summary()

    model.compile(loss='mean_absolute_error',
                    optimizer=Adam(lr=lr, beta_1 = 0.9))





    # Data generator. 
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)

    # Directories for saving models and training stats
    if data_type =="DM":
        model_dir = os.path.join(workspace, "models", "mixdb")
        stats_dir = os.path.join(workspace, "training_stats", "mixdb")
    else:
        model_dir = os.path.join(workspace, "models", "mask_mixdb")
        stats_dir = os.path.join(workspace, "training_stats", "mask_mixdb")
    pp_data.create_folder(model_dir)
    pp_data.create_folder(stats_dir)

    # Print loss before training. 
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

    # Save out training stats. 
    stat_dict = {'iter': iter, 
                    'tr_loss': tr_loss, 
                    'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    # Train. 
    t1 = time.time()

    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        loss = model.train_on_batch(batch_x, batch_y)
        iter += 1
        # Validate and save training stats. 
        if iter % 1000 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
            # Save out training stats. 
            stat_dict = {'iter': iter, 
                        'tr_loss': tr_loss, 
                        'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        # Save model. 
        if iter % 5000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)
        if iter == 70001:
            break
    print("Training time: %s s" % (time.time() - t1,))

def inference(args):
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iter: int, iteration of model to load. 
      visualize: bool, plot enhanced spectrogram for debug. 
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    iter = args.iteration
    data_type = 'IRM'

    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = True
    
    # Load model. 
    if data_type=="DM":
        model_path = os.path.join(workspace, "models", "mixdb", "md_%diters.h5" % 120000)
    else:
        model_path = os.path.join(workspace, "models", "mask_mixdb", "md_%diters.h5" % 265000)
    model = load_model(model_path)
    
    # Load scaler. 
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # Load test data. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "mixdb")
    names = os.listdir(feat_dir)

    for (cnt, na) in enumerate(names):
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_cmplx_x)
        if data_type == "IRM":
            mixed_x = speech_x + noise_x
            mixed_x1 = speech_x + noise_x
        # Process data. 
        n_pad = (n_concat - 1) / 2
        mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
        mixed_x = pp_data.log_sp(mixed_x)

        # Scale data. 
        if scale:
            mixed_x = pp_data.scale_on_2d(mixed_x, scaler)
        
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        
        # Predict. 
        pred = model.predict(mixed_x_3d)
        if data_type =="IRM":
            pred_sp = pred * mixed_x1
        print(cnt, na)
        
        # Inverse scale. 
        if data_type =="DM":
            pred = pp_data.inverse_scale_on_2d(pred, scaler)
            pred_sp = np.exp(pred)
        # Debug plot. 
        # Recover enhanced wav. 
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude 
                                                        # change after spectrogram and IFFT. 
        # Write out enhanced wav. 
        if data_type=="DM":
            out_path = os.path.join(workspace, "enh_wavs", "test", "mixdb", "%s.enh.wav" % na)
        else:
            out_path = os.path.join(workspace, "enh_wavs", "test", "mask_mixdb", "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
        






def continue_train(args):
    workspace = args.workspace
    lr = args.lr
    iter = args.iteration
    data_type = "IRM"
    # Load model. 
    if data_type =="DM":
        model_path = os.path.join(workspace, "models", "mixdb", "md_%diters.h5" % iter)
    else:
        model_path = os.path.join(workspace, "models", "mask_mixdb", "md_%diters.h5" % iter)
    model = load_model(model_path)
    #model = multi_gpu_model(model, 4)
    model.compile(loss='mean_absolute_error',
                    optimizer=Adam(lr=lr, beta_1 = 0.2))
    # Load data. 
    t1 = time.time()
    if data_type=="DM":
        tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "data.h5")
        te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "mixdb" , "data.h5")
    else:
        tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mask_mixdb", "data.h5")
        te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "mask_mixdb" , "data.h5")
    tr_hdf5_dir = os.path.join(workspace, "packed_features", "spectrogram", "train", "mask_mixdb")
    tr_hdf5_names = os.listdir(tr_hdf5_dir)
    tr_hdf5_names = [i for i in tr_hdf5_names if i.endswith(".h5")]
    tr_path_list = [os.path.join(tr_hdf5_dir, i) for i in tr_hdf5_names]
    (tr_x, tr_y) = pp_data.load_hdf5(tr_hdf5_path)
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)
    print("Load data time: %s s" % (time.time() - t1,)) 
    batch_size = 2048
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
    # Scale data. 
    if True:
        t1 = time.time()
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x = pp_data.scale_on_3d(tr_x, scaler)
        te_x = pp_data.scale_on_3d(te_x, scaler)
        if data_type=="DM":
            tr_y = pp_data.scale_on_2d(tr_y, scaler)
            te_y = pp_data.scale_on_2d(te_y, scaler)
        print("Scale data time: %s s" % (time.time() - t1,))
    #scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "scaler.p")
    #scaler = pickle.load(open(scaler_path, 'rb'))
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    #tr_gen = DataGenerator_h5py(batch_size=batch_size, type='train', scaler = scaler)
    #eval_te_gen = DataGenerator_h5py(batch_size=batch_size, type='test', te_max_iter=100, scaler =scaler)
    #eval_tr_gen = DataGenerator_h5py(batch_size=batch_size, type='test', te_max_iter=100, scaler =scaler)
    # Directories for saving models and training stats
    if data_type=="DM":
        model_dir = os.path.join(workspace, "models", "chinese_mixdb", "continue")
        stats_dir = os.path.join(workspace, "training_stats", "chinese_mixdb", "continue")
    else:
        model_dir = os.path.join(workspace, "models", "mask_mixdb", "continue")
        stats_dir = os.path.join(workspace, "training_stats", "mask_mixdb", "continue")   
    pp_data.create_folder(model_dir)
    pp_data.create_folder(stats_dir)
    # Print loss before training. 
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    #tr_loss = eval_h5py(model, eval_tr_gen, tr_path_list)
    #te_loss = eval_h5py(model, eval_te_gen, [te_hdf5_path])
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
    # Save out training stats. 
    stat_dict = {'iter': iter, 
                    'tr_loss': tr_loss, 
                    'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    # Train. 
    t1 = time.time()
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
    #for (batch_x, batch_y) in tr_gen.generate(tr_path_list):
        loss = model.train_on_batch(batch_x, batch_y)
        iter += 1
        # Validate and save training stats. 
        if iter % 500 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            #tr_loss = eval_h5py(model, eval_tr_gen, tr_path_list)
            #te_loss = eval_h5py(model, eval_te_gen, [te_hdf5_path])
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
            # Save out training stats. 
            stat_dict = {'iter': iter, 
                        'tr_loss': tr_loss, 
                        'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        # Save model. 
        if iter % 5000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)
        if iter == 100001:
            break
    print("Training time: %s s" % (time.time() - t1,))




def parser_function(serialized_example):
    features = tf.parse_single_example(serialized_example,
    features={
        'x': tf.FixedLenFeature([], tf.string),
        'y': tf.FixedLenFeature([], tf.string)
        })
    x = tf.reshape(tf.decode_raw(features['x'], tf.float32), [7, 257])
    y = tf.reshape(tf.decode_raw(features['y'], tf.float32), [257,])
    return x, y


def load_tfrecord(batch, repeat, data_path):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(parser_function)
    dataset = dataset.shuffle(buffer_size = 1024*10, seed = 10)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat(repeat)
    iterator = dataset.make_one_shot_iterator()
    tr_x, tr_y  = iterator.get_next()
    return tr_x, tr_y





def continue_train_tfrecord():
    workspace = "workspace"
    lr = 1e-5
    iter = 220000
    data_type = "IRM"
        # Load model. 
    if data_type =="DM":
        model_path = os.path.join(workspace, "models", "elu_mixdb", "md_%diters.h5" % iter)
    else:
        model_path = os.path.join(workspace, "models", "mask_mixdb", "md_%diters.h5" % iter)

    model = load_model(model_path)
    #model = multi_gpu_model(model, 4)
    model.compile(loss='mean_absolute_error',
                    optimizer=Adam(lr=lr, beta_1 = 0.2))
    # Load data. 
    if data_type=="DM":
        tr_hdf5_dir = os.path.join(workspace, "tfrecords", "train", "mixdb")
        tr_hdf5_names = os.listdir(tr_hdf5_dir)
        tr_path_list = [os.path.join(tr_hdf5_dir, i) for i in tr_hdf5_names]
        te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "mixdb", "data.h5")
    else:
        tr_hdf5_dir = os.path.join(workspace, "tfrecords", "train", "mask_mixdb")
        tr_hdf5_names = os.listdir(tr_hdf5_dir)
        tr_path_list = [os.path.join(tr_hdf5_dir, i) for i in tr_hdf5_names]
        te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "mask_mixdb", "data.h5")

    #(tr_x1, tr_y1) = pp_data.load_hdf5("workspace/packed_features/spectrogram/train/mixdb/data100000.h5")
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    t1 = time.time()
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    te_x = pp_data.scale_on_3d(te_x, scaler)
    #tr_x1 = pp_data.scale_on_3d(tr_x1, scaler)
    if data_type=="DM":
        te_y = pp_data.scale_on_2d(te_y, scaler)
        tr_y1 = pp_data.scale_on_2d(tr_y1, scaler)
    print("Scale data time: %s s" % (time.time() - t1,))
    # Directories for saving models and training stats
    if data_type=="DM":
        model_dir = os.path.join(workspace, "models", "elu_mixdb", "continue")
        stats_dir = os.path.join(workspace, "training_stats", "elu_mixdb", "continue")
    else:
        model_dir = os.path.join(workspace, "models", "mask_mixdb", "continue")
        stats_dir = os.path.join(workspace, "training_stats", "mask_mixdb", "continue")   

        pp_data.create_folder(model_dir)
        pp_data.create_folder(stats_dir)
        # Print loss before training. 

        batch_size = 1024*4
        #eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
        eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
        #tr_loss = eval(model, eval_tr_gen, tr_x1, tr_y1)
        tr_loss = 0
        te_loss = eval(model, eval_te_gen, te_x, te_y)
        print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
        # Save out training stats. 
        stat_dict = {'iter': iter, 
                        'tr_loss': tr_loss, 
                        'te_loss': te_loss, }
        stat_path = os.path.join(stats_dir, "%diters.p" % iter)
        cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        # Train. 
        sess = tf.Session()
        x, y = load_tfrecord(batch = batch_size, repeat = 100000, data_path = tr_path_list)
        t1 = time.time()
        for count in range(1000000000):
            [tr_x, tr_y] = sess.run([x, y])
            loss = model.train_on_batch(tr_x, tr_y)
            iter += 1
            # Validate and save training stats. 
            if iter % 1000 == 0:
                #tr_loss = eval(model, eval_tr_gen, tr_x1, tr_y1)
                te_loss = eval(model, eval_te_gen, te_x, te_y)
                print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
                # Save out training stats. 
                stat_dict = {'iter': iter, 
                            'tr_loss': tr_loss, 
                            'te_loss': te_loss, }
                stat_path = os.path.join(stats_dir, "%diters.p" % iter)
                cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
            # Save model. 
            if iter % 5000 == 0:
                model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
                model.save(model_path)
                print("Saved model to %s" % model_path)
            if iter == 100001:
                break
        print("Training time: %s s" % (time.time() - t1,))








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, required=True)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    
    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    else:
        raise Exception("Error!")












