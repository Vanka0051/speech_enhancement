import os
import shutil

timit_path = '../../database/TIMIT/TEST/'
def rename_and_move_wavfile(timit_path):
    tmp = timit_path.split("/")[-2]
    if tmp=="TEST":
        target_path_base = './mini_data/test_speech/'
    elif tmp=="TRAIN":
        target_path_base = './mini_data/train_speech/'
    else:
        print("input path error")
        return 0
    for root, dirs, files in os.walk(timit_path):
        for file in files:
            curr_path = os.path.join(root, file)
            suffix = os.path.splitext(curr_path)[-1]
            if suffix == ".WAV":
                train_type = curr_path.split("/")[-4]
                district_type = curr_path.split("/")[-3]
                speeker_id = curr_path.split("/")[-2]
                sentence_id = curr_path.split("/")[-1]
                target_path = target_path_base + train_type + "_" + \
                district_type+ "_" +speeker_id+ "_" +sentence_id
                shutil.copy(curr_path, target_path)
