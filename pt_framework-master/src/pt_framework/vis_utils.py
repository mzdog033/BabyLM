import numpy as np
import matplotlib.pyplot as plt
import pylab
import json
import pdb
import copy

import os
from matplotlib.backends.backend_pdf import PdfPages


def get_or_load(
        cache_dict, conn, dbname, 
        colname, curr_expid, refresh_cache,
        from_folder=None, remove_train_results=False):
    cache_key = os.path.join(dbname, colname, curr_expid)
    if not remove_train_results:
        cache_key += '_$wt'
    if refresh_cache or cache_key not in cache_dict:
        min_step = -1
        if len(cache_dict.get(cache_key, [])) > 0:
            min_step = cache_dict[cache_key][-1]['step']
        if from_folder is None:
            find_res = conn[dbname][colname].find(
                        {'exp_id': curr_expid, 
                         'train_results': {'$exists': True},
                         'step': {'$gt': min_step}})
        else:
            from_folder = os.path.join(from_folder, 'records')
            record_paths = os.listdir(from_folder)
            def iter_filter_func(_path):
                iter_num = _path.split('_')[1]
                iter_num = int(iter_num[4:])
                return iter_num > min_step
            record_paths = list(filter(iter_filter_func, record_paths))
            record_paths = [
                    os.path.join(from_folder, _path)
                    for _path in record_paths]
            find_res = [json.load(open(_path, 'r')) for _path in record_paths]
        find_res = list(sorted(find_res, key = lambda x: x['step']))
        find_res = list(filter(lambda x: x['step'] > min_step, find_res))
        if len(find_res) > 0:
            new_find_res = []
            for curr_indx in range(len(find_res)-1):
                if find_res[curr_indx]['step'] == find_res[curr_indx+1]['step']:
                    continue
                new_find_res.append(find_res[curr_indx])
            new_find_res.append(find_res[len(find_res)-1])
            find_res = new_find_res
        if remove_train_results:
            for each_res in find_res:
                if 'train_results' in each_res:
                    del each_res['train_results']
        if cache_key in cache_dict:
            cache_dict[cache_key].extend(find_res)
        else:
            cache_dict[cache_key] = find_res
        find_res = cache_dict[cache_key]
    else:
        find_res = cache_dict[cache_key]
    return find_res


def show_train(
        curr_expid, 
        cache_dict,
        conn,
        dbname='flamingo', 
        colname='cc12m', 
        start_N=50, 
        batch_watch_start=0,
        batch_watch_end=None,
        do_conv=False, 
        conv_len=100, 
        new_figure=True, 
        batch_size=8, 
        batch_offset=0, 
        max_step=None, 
        label_now=None,
        loss_key='loss',
        refresh_cache=True,
        from_folder=None,
        loss_mult=None,
        figsize=(9,5),
        ):
    if label_now is None:
        label_now = curr_expid

    find_res = get_or_load(
            cache_dict, conn, dbname, colname, curr_expid, refresh_cache,
            from_folder)
    if max_step:
        find_res = filter(lambda x: x['step']<max_step, find_res)
    if len(find_res) == 0:
        return
    find_res = filter(lambda x: 'train_results' in x, find_res)
    if isinstance(loss_key, str):
        _take_loss = lambda _r: _r[loss_key]
    else:
        _take_loss = loss_key
    train_vec = np.concatenate(
                [[_take_loss(_r) for _r in r['train_results']] 
                for r in find_res])

    _N = start_N
    if new_figure:
        fig = plt.figure(figsize=figsize)
    inter_list = train_vec[_N:]
    inter_list = np.asarray(inter_list)
    inter_list = inter_list[inter_list < 50]
    if do_conv:
        conv_list = np.ones([conv_len])/conv_len
        inter_list = np.convolve(inter_list, conv_list, mode='valid')
    
    temp_x_list = np.asarray(range(len(inter_list)))*1.0*batch_size/(10000*8) + batch_offset
    new_indx_list = temp_x_list > batch_watch_start
    if batch_watch_end is not None:
        new_indx_list = (temp_x_list>batch_watch_start) & (temp_x_list<batch_watch_end)
    if loss_mult is not None:
        assert isinstance(loss_mult, int)
        inter_list *= loss_mult
    plt.plot(temp_x_list[new_indx_list], inter_list[new_indx_list], label = label_now)
    plt.title('Training loss')
    plt.legend(loc = 'best')
