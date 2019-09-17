import numpy as np
import h5py
import prepare_data as pp_data
class DataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=None):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        
    def generate(self, xs, ys):
        x = xs[0]
        y = ys[0]
        batch_size = self._batch_size_
        n_samples = len(x)
        
        index = np.arange(n_samples)
        np.random.shuffle(index)
        
        iter = 0
        epoch = 0
        pointer = 0
        while True:
            if (self._type_ == 'test') and (self._te_max_iter_ is not None):
                if iter == self._te_max_iter_:
                    break
            iter += 1
            if pointer >= n_samples:
                epoch += 1
                if (self._type_) == 'test' and (epoch == 1):
                    break
                pointer = 0
                np.random.shuffle(index)                
 
            batch_idx = index[pointer : min(pointer + batch_size, n_samples)]
            pointer += batch_size
            yield x[batch_idx], y[batch_idx]





class DataGenerator_h5py(object):
    def __init__(self,  batch_size, type, scaler, te_max_iter=None , ):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        self._scaler_ = scaler
        
    def generate(self, path_list):
        iter = 0
        epoch = 0
        pointer = 0
        path = path_list[epoch]
        n_file = len(path_list)
        data = h5py.File(path)
        x = data['x']
        y = data['y']
        batch_size = self._batch_size_
        n_samples = len(x)
        index = np.arange(n_samples)
        np.random.shuffle(index)
        while True:
            if (self._type_ == 'test') and (self._te_max_iter_ is not None):
                if iter == self._te_max_iter_:
                    break
            iter += 1
            if pointer >= n_samples:
                epoch += 1
                if epoch == n_file:
                    epoch = 0
                path = path_list[epoch]
                print("start %s"%path)
                n_file = len(path_list)
                data = h5py.File(path)
                x = data['x']
                y = data['y']
                if (self._type_) == 'test' and (epoch == n_file - 1):
                    break
                pointer = 0
                np.random.shuffle(index)                
 
            batch_idx = index[pointer : min(pointer + batch_size, n_samples)]
            pointer += batch_size
            yield pp_data.scale_on_3d(x[sorted(batch_idx)], self._scaler_), pp_data.scale_on_2d(y[sorted(batch_idx)], self._scaler_)



'''
count = 0
tr_gen = DataGenerator_h5py(batch_size = 10, type = "train")
for (batch_x, batch_y) in tr_gen.generate(path_list = ["data1.h5", "data2.h5"]):
    count+=1
    print(count)
'''


