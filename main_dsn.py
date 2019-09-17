from main_dnn import *
from keras.layers import Reshape, Input, Concatenate, Dense, Lambda
import keras


def lambda_slice(x, n):
    return x[:, n, :]

def continue_train_tfrecord():

    data_type = "IRM"
    workspace = "workspace"
    lr = 1e-4
    input_x = Input(shape = (7, 257))
    l1_0 = Lambda(lambda_slice, arguments = {"n": 0})(input_x)
    l1 = Dense(1024, activation=None)(l1_0)
    l1_1 = Dense(10, activation='elu')(l1)
    l2_0 = Lambda(lambda_slice, arguments = {"n": 1})(input_x)
    l2_input = Concatenate(axis = -1)([l1_1, l2_0])
    l2 = Dense(1024, activation=None)(l2_input)
    l2_1 = Dense(10, activation='elu')(l2)
    l3_0 = Lambda(lambda_slice, arguments = {"n": 2})(input_x)
    l3_input = Concatenate(axis = -1)([l1_1, l2_1, l3_0])
    l3 = Dense(1024, activation=None)(l3_input)
    l3_1 = Dense(10, activation='elu')(l3)
    l4_0 = Lambda(lambda_slice, arguments = {"n": 3})(input_x)
    l4_input = Concatenate(axis = -1)([l1_1, l2_1, l3_1, l4_0])
    l4 = Dense(1024, activation=None)(l4_input)
    l4_1 = Dense(10, activation='elu')(l4)
    l5_0 = Lambda(lambda_slice, arguments = {"n": 4})(input_x)
    l5_input = Concatenate(axis = -1)([l1_1, l2_1, l3_1, l4_1, l5_0])
    l5 = Dense(1024, activation=None)(l5_input)
    l5_1 = Dense(10, activation='elu')(l5)
    l6_0 = Lambda(lambda_slice, arguments = {"n": 5})(input_x)
    l6_input = Concatenate(axis = -1)([l1_1, l2_1, l3_1, l4_1, l5_1, l6_0])
    l6 = Dense(1024, activation=None)(l6_input)
    l6_1 = Dense(10, activation='elu')(l6)
    l7_0 = Lambda(lambda_slice, arguments = {"n": 6})(input_x)
    l7_input = Concatenate(axis = -1)([l1_1, l2_1, l3_1, l4_1, l5_1, l7_0])
    outputs = Dense(257, activation=None)(l7_input)


    model = keras.models.Model(inputs = [input_x], outputs = outputs)
    model.compile(loss='mean_absolute_error',
                    optimizer=Adam(lr=lr, beta_1 = 0.9))
    # Load data. 
    tr_hdf5_dir = os.path.join(workspace, "tfrecords", "train", "mixdb")
    tr_hdf5_names = os.listdir(tr_hdf5_dir)
    tr_path_list = [os.path.join(tr_hdf5_dir, i) for i in tr_hdf5_names]
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "mixdb", "data.h5")
    
    (tr_x1, tr_y1) = pp_data.load_hdf5("workspace/packed_features/spectrogram/train/mixdb/data100000.h5")
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    t1 = time.time()
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "mixdb", "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    te_x = pp_data.scale_on_3d(te_x, scaler)
    tr_x1 = pp_data.scale_on_3d(tr_x1, scaler)
    te_y = pp_data.scale_on_2d(te_y, scaler)
    tr_y1 = pp_data.scale_on_2d(tr_y1, scaler)
    print("Scale data time: %s s" % (time.time() - t1,))
    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "dsn_mixdb", "continue")
    stats_dir = os.path.join(workspace, "training_stats", "elu_mixdb", "continue")

    pp_data.create_folder(model_dir)
    pp_data.create_folder(stats_dir)
    # Print loss before training. 
    iter = 0
    batch_size = 1024
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    tr_loss = eval(model, eval_tr_gen, tr_x1, tr_y1)
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
            tr_loss = eval(model, eval_tr_gen, tr_x1, tr_y1)
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


def inference(args):
    workspace = "workspace"
    n_concat = 7
    iter = args.iteration
    n_window = 512
    n_overlap = 256
    fs = 16000
    # Load model. 
    model_path = os.path.join(workspace, "models", "dsn_mixdb", "md_%diters.h5" % iter)
    model = load_model(model_path)
    # Load test data. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "dsn_mixdb")
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
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)#[100, 7, 257]
        #mixed_x = pad_with_border(mixed_x, n_pad)
        #mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        # Predict. 
        w, h, l = mixed_x_3d.shape 
        pred = model.predict(mixed_x_3d)
        #pred_sp = pred[:, -1, :]
        print(cnt, na)
        if False:
            pred_sp = np.load("pred_sp.npy")
            speech_x = np.load("speech_x.npy")
            mixed_x = np.load("mixed_x.npy")
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
        out_path = os.path.join(workspace, "enh_wavs", "test", "dsn_mixdb", "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
  