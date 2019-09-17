"""
Summary:  Calculate PESQ and overal stats of enhanced speech. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import argparse
import os
import csv
import numpy as np
import cPickle
import soundfile
from pypesq import pypesq
from pystoi.stoi import stoi
from prepare_data import create_folder
#import matplotlib.pyplot as plt


def plot_training_stat(args):
    """Plot training and testing loss. 
    
    Args: 
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      bgn_iter: int, plot from bgn_iter
      fin_iter: int, plot finish at fin_iter
      interval_iter: int, interval of files. 
    """
    workspace = args.workspace
    tr_snr = args.tr_snr
    bgn_iter = args.bgn_iter
    fin_iter = args.fin_iter
    interval_iter = args.interval_iter

    tr_losses, te_losses, iters = [], [], []
    
    # Load stats. 
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    for iter in xrange(bgn_iter, fin_iter, interval_iter):
        stats_path = os.path.join(stats_dir, "%diters.p" % iter)
        dict = cPickle.load(open(stats_path, 'rb'))
        tr_losses.append(dict['tr_loss'])
        te_losses.append(dict['te_loss'])
        iters.append(dict['iter'])
        
    # Plot
#    line_tr, = plt.plot(tr_losses, c='b', label="Train")
#    line_te, = plt.plot(te_losses, c='r', label="Test")
#    plt.axis([0, len(iters), 0, max(tr_losses)])
#    plt.xlabel("Iterations")
#    plt.ylabel("Loss")
#    plt.legend(handles=[line_tr, line_te])
#    plt.xticks(np.arange(len(iters)), iters)
#    plt.show()


def calculate_pesq(args):
    """Calculate PESQ of all enhaced speech. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of clean speech. 
      te_snr: float, testing SNR. 
    """
    # Remove already existed file. 
    data_type = args.data_type
    speech_dir = "mini_data/test_speech"
    f = "{0:<16} {1:<16} {2:<16}"
    print(f.format("0", "Noise", "PESQ"))
    f1 = open(data_type + '_pesq_results.csv', 'w')
    f1.write("%s\t%s\n"%("audio_id", "PESQ"))
    # Calculate PESQ of all enhaced speech. 
    if data_type=="DM":
        enh_speech_dir = os.path.join("workspace", "enh_wavs", "test", "mixdb")
    elif data_type=="IRM":
        enh_speech_dir = os.path.join("workspace", "enh_wavs", "test", "mask_mixdb")
    elif data_type=="CRN":
        enh_speech_dir = os.path.join("workspace", "enh_wavs", "test", "crn_mixdb")
    elif data_type=="PHASE":
        enh_speech_dir = os.path.join("workspace", "enh_wavs", "test", "phase_spec_clean_mixdb")
    elif data_type=="VOLUME":
        enh_speech_dir = os.path.join("workspace", "enh_wavs", "test", "volume_mixdb")
    elif data_type=="NOISE":
        enh_speech_dir = os.path.join("workspace" ,'mixed_audios','spectrogram','test','mixdb')
    names = os.listdir(enh_speech_dir)
    for (cnt, na) in enumerate(names):
        enh_path = os.path.join(enh_speech_dir, na)
        enh_audio, fs = soundfile.read(enh_path)
        speech_na = na.split('.')[0]
        speech_path = os.path.join(speech_dir, "%s.WAV" % speech_na)
        speech_audio, fs = soundfile.read(speech_path)
        #alpha = 1. / np.max(np.abs(speech_audio))
        #speech_audio *=alpha
        pesq_ = pypesq(16000, speech_audio, enh_audio, 'wb') 
        print(f.format(cnt, na, pesq_))
        f1.write("%s\t%f\n"%(na, pesq_))
        # Call executable PESQ tool. 
        #cmd = ' '.join(["./pesq", speech_path, enh_path, "+16000"])
        #os.system(cmd)        
    os.system("mv %s_pesq_results.csv ./pesq_result/%s_pesq_results.csv"%(data_type, data_type))

        
def get_stats(args):
    """Calculate stats of PESQ. 
    """
    data_type = args.data_type
    pesq_path = "./pesq_result/"+ data_type+ "_pesq_results.csv"
    with open(pesq_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
    pesq_dict = {}
    for i1 in xrange(1, len(lis) - 1):
        li = lis[i1]
        na = li[0]
        pesq = float(li[1])
        noise_type = na.split('.')[1]
        if noise_type not in pesq_dict.keys():
            pesq_dict[noise_type] = [pesq]
        else:
            pesq_dict[noise_type].append(pesq)
    out_csv_path ='./pesq_result/'+ data_type +'_pesq_differentnoise.csv'
    csv_file = open(out_csv_path, 'w')
    avg_list, std_list = [], []
    f = "{0:<16} {1:<16}"
    print(f.format("Noise", "PESQ"))
    csv_file.write("%s\t%s\n"%("Noise", "PESQ"))
    print("---------------------------------")
    csv_file.write("----------------\t-----------------\n")
    for noise_type in pesq_dict.keys():
        pesqs = pesq_dict[noise_type]
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        avg_list.append(avg_pesq)
        std_list.append(std_pesq)
        print(f.format(noise_type, "%.2f +- %.2f" % (avg_pesq, std_pesq)))
        csv_file.write("%s\t%s\n"%(noise_type, "%.2f +- %.2f" % (avg_pesq, std_pesq)))
    print("---------------------------------")
    csv_file.write("----------------\t-----------------\n")
    print(f.format("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list))))
    csv_file.write("%s\t%s\n"%("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list))))
    csv_file.close()





def get_snr_stats(args):

    data_type = args.data_type
    pesq_path = os.path.join("pesq_result", data_type + "_pesq_results.csv")
    with open(pesq_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        pesq_lis = list(reader)

    pesq_lis[0].append("SNR")
    pesq_title = pesq_lis[0]
    pesq_lis = pesq_lis[:-1]
    csv_path = os.path.join("workspace", "mixture_csvs", "test_1hour_even.csv")
    with open(csv_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        csv_lis = list(reader)

    count = 0
    for csv_name in csv_lis[1:]:
        if data_type=="NOISE":
            csv_na = csv_name[0].split(".")[0] + "." + csv_name[1].split(".")[0]+ "."+csv_name[-1] + "db.wav"
        else:
            csv_na = csv_name[0].split(".")[0] + "." + csv_name[1].split(".")[0]+ "."+csv_name[-1] + "db.enh.wav"
        for pesq_name in pesq_lis[1:]:
            if csv_na == pesq_name[0]:
                count+=1
                pesq_name.append(csv_name[-1])
                break

    pesq_dict = {}
    for i1 in xrange(1, len(pesq_lis)):
        li = pesq_lis[i1]
        na = li[0]
        pesq = float(li[1][0:4])
        snr = float(li[-1])
        snr_key = snr
        if snr_key not in pesq_dict.keys():
            pesq_dict[snr_key] = [pesq]
        else:
            pesq_dict[snr_key].append(pesq)

    out_csv_path = os.path.join( "pesq_result", data_type + "_snr_results.csv")
    create_folder(os.path.dirname(out_csv_path))
    csv_file = open(out_csv_path, 'w')
    avg_list, std_list = [], []
    sample_sum = 0
    f = "{0:<16} {1:<16} {2:<16}"
    print(f.format("SNR", "PESQ", "SAMPLE_NUM"))
    csv_file.write("%s\t%s\t%s\n"%("SNR", "PESQ", "SAMPLE_NUM"))
    csv_file.flush()
    print("---------------------------------")
    for snr_type in sorted(pesq_dict.keys()):
        pesqs = pesq_dict[snr_type]
        sample_num = len(pesqs)
        sample_sum+=sample_num
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        avg_list.append(avg_pesq)
        std_list.append(std_pesq)
        print(f.format(snr_type, "%.2f +- %.2f" % (avg_pesq, std_pesq), sample_num))
        csv_file.write("%s\t%s\t%s\n"%(snr_type, "%.2f +- %.2f" % (avg_pesq, std_pesq), sample_num))
        csv_file.flush()

    print("---------------------------------")
    print(f.format("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list)), sample_sum))
    csv_file.write("%s\t%s\t%s\n"%("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list)), "%d"%sample_sum))
    csv_file.close()





def calculate_stoi(args):
    workspace = "workspace"
    speech_dir = "mini_data/test_speech"
    # Calculate PESQ of all enhaced speech. 
    enh_speech_dir = os.path.join(workspace, "enh_wavs", "test", "mixdb")
    #enh_speech_dir = "/data00/wangjinchao/sednn-master/mixture2clean_dnn/workspace/mixed_audios/spectrogram/test/mixdb"
    #    enh_speech_dir = os.path.join(workspace ,'mixed_audios','spectrogram','test','mixdb')
    names = os.listdir(enh_speech_dir)
    f = open("IRM_stoi.txt", "w")
    f.write("%s\t%s\n"%("speech_id", "stoi"))
    f.flush()
    for (cnt, na) in enumerate(names):
        print(cnt, na)
        enh_path = os.path.join(enh_speech_dir, na)
        speech_na = na.split('.')[0]
        speech_path = os.path.join(speech_dir, "%s.WAV" % speech_na)
        speech_audio, fs = read_audio(speech_path, 16000)
        enhance_audio, fs = read_audio(enh_path, 16000)
        if len(speech_audio)>len(enhance_audio):
            speech_audio = speech_audio[:len(enhance_audio)]
        else:
            enhance_audio = enhance_audio[:len(speech_audio)]
        stoi_value = stoi(speech_audio, enhance_audio, fs, extended = False)
        f.write("%s\t%f\n"%(na, stoi_value))
        f.flush()
    f.close()
        






def get_stoi_stats(args):
    stoi_path = "./stoi_result/IRM_stoi.txt"
    with open(stoi_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)

    stoi_dict = {}
    for i1 in xrange(1, len(lis) - 1):
        li = lis[i1]
        na = li[0]
        stoi = float(li[1])
        noise_type = na.split('.')[1]
        if noise_type not in stoi_dict.keys():
            stoi_dict[noise_type] = [stoi]
        else:
            stoi_dict[noise_type].append(stoi)
        #out_csv_path ='./stoi_result/gvdm_enhance.csv'
        #csv_file = open(out_csv_path, 'w')
    avg_list, std_list = [], []
    f = "{0:<16} {1:<16}"
    print(f.format("Noise", "STOI"))
    #csv_file.write("%s\t%s\n"%("Noise", "stoi"))
    print("---------------------------------")
    #csv_file.write("----------------\t-----------------\n")
    for noise_type in stoi_dict.keys():
        stois = stoi_dict[noise_type]
        avg_stoi = np.mean(stois)
        std_stoi = np.std(stois)
        avg_list.append(avg_stoi)
        std_list.append(std_stoi)
        print(f.format(noise_type, "%.5f +- %.5f" % (avg_stoi, std_stoi)))
        #csv_file.write("%s\t%s\n"%(noise_type, "%.2f +- %.2f" % (avg_stoi, std_stoi)))
    print("---------------------------------")
    #csv_file.write("----------------\t-----------------\n")
    print(f.format("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list))))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_plot_training_stat = subparsers.add_parser('plot_training_stat')
    parser_plot_training_stat.add_argument('--workspace', type=str, required=True)
    parser_plot_training_stat.add_argument('--tr_snr', type=float, required=True)
    parser_plot_training_stat.add_argument('--bgn_iter', type=int, required=True)
    parser_plot_training_stat.add_argument('--fin_iter', type=int, required=True)
    parser_plot_training_stat.add_argument('--interval_iter', type=int, required=True)

    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--data_type', type=str, required=True)
    
    parser_get_stats = subparsers.add_parser('get_stats')
    parser_get_stats.add_argument('--data_type', type=str, required=True)
    
    parser_get_snr_stats = subparsers.add_parser('get_snr_stats')
    parser_get_snr_stats.add_argument('--data_type', type=str, required=True)



    args = parser.parse_args()
    
    if args.mode == 'plot_training_stat':
        plot_training_stat(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    elif args.mode == 'get_stats':
        get_stats(args)
    elif args.mode == 'get_snr_stats':
        get_snr_stats(args)
    else:
        raise Exception("Error!")
