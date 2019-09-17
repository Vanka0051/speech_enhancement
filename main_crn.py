from main_dnn import *
from keras.layers import Reshape, Conv2D, BatchNormalization, ZeroPadding2D, Lambda
from keras.layers import Input, Concatenate, LSTM, Conv2DTranspose, Cropping2D, ELU
import keras
import tensorflow as tf
import time
import os
import config as cfg
import prepare_data as pp_data
from spectrogram_to_wave import *


def parser_function(serialized_example):
    features = tf.parse_single_example(serialized_example,
    features={
        'x': tf.FixedLenFeature([], tf.string),
        'y': tf.FixedLenFeature([], tf.string)
        })
    x = tf.reshape(tf.decode_raw(features['x'], tf.float32), [11, 161])
    y = tf.reshape(tf.decode_raw(features['y'], tf.float32), [11, 161])
    return x, y


def load_tfrecord(batch, repeat, data_path):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(parser_function)
    dataset = dataset.shuffle(10240)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat(repeat)
    iterator = dataset.make_one_shot_iterator()
    tr_x, tr_y  = iterator.get_next()
    return tr_x, tr_y



def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x]
    return np.concatenate(x_pad_list, axis=0)


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)







def inference(args):
    workspace = "workspace"
    n_concat = 11
    iter = 50000
    n_window = 320
    n_overlap = 160
    fs = 16000
    # Load model. 
    model_path = os.path.join(workspace, "models", "crn_mixdb", "md_%diters.h5" % iter)
    model = load_model(model_path, custom_objects={'keras': keras})
    # Load test data. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "crn_mixdb")
    #feat_dir = os.path.join(workspace, "features", "spectrogram", "train", "office_mixdb")
    names = os.listdir(feat_dir)
    for (cnt, na) in enumerate(names):
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_cmplx_x)
        # Process data. 
        n_pad = (n_concat - 1) 
        #mixed_x = pad_with_border(mixed_x, n_pad)
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=11)#[100, 7, 257]
        #mixed_x = pad_with_border(mixed_x, n_pad)
        #mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        # Predict. 
        w, h, l = mixed_x_3d.shape 
        pred = model.predict(mixed_x_3d)
        pred_sp = np.reshape(pred, [w*h, l])
        mixed_cmplx_x = mixed_cmplx_x[:w*h, :]
        #pred_sp = pred[:, -1, :]
        print(cnt, na)
        if False:
            fig, axs = plt.subplots(3,1, sharex=False)
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred_sp.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("%ddb mixture log spectrogram" % int(1))
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in range(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            plt.show()
            # Recover enhanced wav. 
        #pred_sp = np.exp(pred)
        #pred_sp = pred
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude 
        # Write out enhanced wav. 
        out_path = os.path.join(workspace, "enh_wavs", "test", "crn_mixdb", "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
    







def train_tfrecords(args):
    lr = args.lr
    # Load data. 
    t1 = time.time()
    tr_hdf5_dir = os.path.join("workspace", "tfrecords", "train", "crn_mixdb")
    tr_hdf5_names = os.listdir(tr_hdf5_dir)
    tr_path_list = [os.path.join(tr_hdf5_dir, i) for i in tr_hdf5_names]
    te_hdf5_path = os.path.join("workspace", "packed_features", "spectrogram", "test", "crn_mixdb" , "data.h5")
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    print("test.h5 loaded ! ! !")
    train_path = os.path.join("workspace", "packed_features", "spectrogram", "train", "crn_mixdb" , "data.h5")
    (tr_x, tr_y) = pp_data.load_hdf5(train_path)
    print("train.h5 loaded ! ! !")
    batch_size = 1024

    # Scale data. 
    t1 = time.time()
        
    input_x = Input(shape = (11, 161))
    reshape_x = Reshape((1, 11, 161), input_shape = (11, 161))(input_x)
    l1_input = ZeroPadding2D(padding = ((1, 0), (0, 0)), data_format = "channels_first")(reshape_x)
    l1 = Conv2D(filters=16,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l1_input)
    l1 = BatchNormalization()(l1)
    l1 = ELU()(l1)

    l2_input = ZeroPadding2D(padding = ((1, 0), (0, 0)), data_format = "channels_first")(l1)
    l2 = Conv2D(filters=32,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first" , padding = "valid")(l2_input)
    l2 = BatchNormalization()(l2)
    l2 = ELU()(l2)

    l3_input = ZeroPadding2D(padding = ((1, 0), (0, 0)), data_format = "channels_first")(l2)
    l3 = Conv2D(filters=64,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l3_input)
    l3 = BatchNormalization()(l3)
    l3 = ELU()(l3)

    l4_input = ZeroPadding2D(padding = ((1, 0), (0, 0)), data_format = "channels_first")(l3)
    l4 = Conv2D(filters=128,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l4_input)
    l4 = BatchNormalization()(l4)
    l4 = ELU()(l4)

    l5_input = ZeroPadding2D(padding = ((1, 0), (0, 0)), data_format = "channels_first")(l4)
    l5 = Conv2D(filters=256,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l5_input)
    l5 = BatchNormalization()(l5)
    l5 = ELU()(l5)

    reshape_x2 = Reshape((11, 4*256), input_shape = (11, 4, 256))(l5)
    lstm1 = LSTM(units = 4*256, activation = 'tanh', return_sequences  = True)(reshape_x2)
    lstm2 = LSTM(units = 4*256, activation = 'tanh', return_sequences  = True)(lstm1)
    reshape_x3 = Reshape((256, 11, 4), input_shape = (11, 4*256))(lstm2)


    l8_input = Concatenate(axis = 1)([reshape_x3, l5])
    l8 = Conv2DTranspose(filters=128,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l8_input)
    l8 = Cropping2D(cropping = ((1, 0), (0, 0)), data_format = "channels_first")(l8)
    l8 = BatchNormalization()(l8)
    l8 = ELU()(l8)


    l9_input = Concatenate(axis = 1)([l8, l4])
    l9 = Conv2DTranspose(filters=64,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l9_input)
    l9 = Cropping2D(cropping = ((1, 0), (0, 0)), data_format = "channels_first")(l9)
    l9 = BatchNormalization()(l9)
    l9 = ELU()(l9)


    l10_input = Concatenate(axis = 1)([l9, l3])
    l10 = Conv2DTranspose(filters=32,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l10_input)
    l10 = Cropping2D(cropping = ((1, 0), (0, 0)), data_format = "channels_first")(l10)
    l10 = BatchNormalization()(l10)
    l10 = ELU()(l10)

    l11_input = Concatenate(axis = 1)([l10, l2])
    l11_input = ZeroPadding2D(padding = ((0, 0), (1, 0)), data_format = "channels_first")(l11_input)
    l11 = Conv2DTranspose(filters=16,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l11_input)
    l11 = Cropping2D(cropping = ((1, 0), (1, 0)), data_format = "channels_first")(l11)
    l11 = BatchNormalization()(l11)
    l11 = ELU()(l11)

    l12_input = Concatenate(axis = 1)([l11, l1])
    l12 = Conv2DTranspose(filters=1,kernel_size=(2,3),strides=(1,2), activation=None
    , data_format="channels_first", padding = "valid")(l12_input)
    l12 = Cropping2D(cropping = ((1, 0), (0, 0)), data_format = "channels_first")(l12)
    l12 = Reshape((11, 161), input_shape = (11, 161, 1))(l12)
    l12 = Lambda(lambda x: keras.activations.softplus(x))(l12)
    #l12 = keras.layers.Lambda(lambda x:keras.activations.softplus(x))(l12)
    model = keras.models.Model(inputs = [input_x], outputs = l8)
    model.summary()
    #lr = 5e-5
    #model_path = os.path.join(workspace, "models", "crn_mixdb", "md_%diters.h5" % 3935)
    #model = load_model(model_path, custom_objects={'tf': tf})
    #model = multi_gpu_model(model, 4)

    model.compile(loss='mean_absolute_error',
                    optimizer=Adam(lr=lr, beta_1 = 0.9))
    print("model is built ! ! !")
    # Data generator. 
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)

    # Directories for saving models and training stats
    model_dir = os.path.join("workspace", "models", "crn_mixdb")
    pp_data.create_folder(model_dir)

    stats_dir = os.path.join("workspace", "training_stats", "crn_mixdb")
    pp_data.create_folder(stats_dir)

    # Print loss before training. 
    iter = 0
    print("start calculating initial loss.......")
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
    sess = tf.Session()
    x, y = load_tfrecord(batch = batch_size, repeat = 100000, data_path = tr_path_list)
    t1 = time.time()
    for count in range(1000000000):
        [tr_x, tr_y] = sess.run([x, y])
        loss = model.train_on_batch(tr_x, tr_y)
        iter += 1
        # Validate and save training stats. 
        if iter % 1000 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            #te_loss = tr_loss
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
    print("Training time: %s s" % (time.time() - t1,))







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--lr', default = 1e-4, type=float, required=False)

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--lr', default = 1e-4, type=float, required=False)
    parser_inference.add_argument('--iteration', type=int, default=50000)
    args = parser.parse_args()

    if args.mode=="inference":
        inference(args)
    else:
        train(args)





