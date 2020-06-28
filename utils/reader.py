import pickle as pickle
import numpy as np

def unison_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data['y'])
    c1 = np.array(data['c1'])
    c2 = np.array(data['c2'])
    r = np.array(data['r'])

    assert len(y) == len(c1) == len(c2) == len(r)
    p = np.random.permutation(len(y))
    shuffle_data = {'y': y[p], 'c1': c1[p], 'c2': c2[p], 'r': r[p]}
    return shuffle_data

def split_c(c, split_id):
    '''c is a list, example context
       split_id is a integer, conf[_EOS_]
       return nested list
    '''
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns

def normalize_length(_list, length, cut_type='tail'):
    '''_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    '''
    real_length = len(_list)
    if real_length == 0:
        return [0]*length, 0 # hh

    if real_length <= length:
        if not isinstance(_list[0], list):
            _list.extend([0]*(length - real_length))
        else:
            _list.extend([[0]]*(length - real_length)) # hh
        return _list, real_length

    if cut_type == 'head':
        return _list[:length], length
    if cut_type == 'tail':
        return _list[-length:], length


def produce_one_sample(data, index, split_id, max_turn_num, max_turn_len, turn_cut_type='tail', term_cut_type='tail'):
    '''max_turn_num=10
       max_turn_len=50
       return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
    '''
    c1 = data['c1'][index]
    c2 = data['c2'][index]
    r = data['r'][index]
    y = data['y'][index]

    turns1 = split_c(c1, split_id)
    #normalize turns_c length, nor_turns length is max_turn_num
    assert len(turns1)
    nor_turns1, turn_len1 = normalize_length(turns1, max_turn_num, turn_cut_type)

    nor_turns_nor_c1 = []
    term_len1 = []
    #nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
    for c in nor_turns1:
        #nor_c length is max_turn_len
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c1.append(nor_c)
        term_len1.append(nor_c_len)

    turns2 = split_c(c2, split_id)
    #normalize turns_c length, nor_turns length is max_turn_num
    assert len(turns2)
    nor_turns2, turn_len2 = normalize_length(turns2, max_turn_num, turn_cut_type)

    nor_turns_nor_c2 = []
    term_len2 = []
    #nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
    for c in nor_turns2:
        #nor_c length is max_turn_len
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c2.append(nor_c)
        term_len2.append(nor_c_len)
    try:
        #print("========================\n", r)
        r = [int(i) for i in r]
        nor_turns_nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)
        #print(r)
    except Exception:
        #print(Exception)
        print(r)
        #nor_turns_nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)
        exit()

    return y, nor_turns_nor_c1, nor_turns_nor_c2, nor_turns_nor_r, turn_len1, term_len1, turn_len2, term_len2, r_len


def build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns1 = []
    _turns2 = []
    _tt_turns_len1 = []
    _every_turn_len1 = []
    _tt_turns_len2 = []
    _every_turn_len2 = []

    _response = []
    _response_len = []

    _label = []

    for i in range(conf['batch_size']):
        index = batch_index * conf['batch_size'] + i
        y, nor_turns_nor_c1, nor_turns_nor_c2, nor_r, turn_len1, term_len1, turn_len2, term_len2, r_len = produce_one_sample(data, index, conf['_EOS_'], conf['max_turn_num'],
                conf['max_turn_len'], turn_cut_type, term_cut_type)

        _label.append(y)
        _turns1.append(nor_turns_nor_c1)
        _turns2.append(nor_turns_nor_c2)
        _response.append(nor_r)
        _every_turn_len1.append(term_len1)
        _tt_turns_len1.append(turn_len1)
        _every_turn_len2.append(term_len2)
        _tt_turns_len2.append(turn_len2)
        _response_len.append(r_len)

    return _turns1, _turns2, _tt_turns_len1, _every_turn_len1,  _tt_turns_len2, _every_turn_len2, _response, _response_len, _label

def build_one_batch_dict(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail'):
    assert False
    _turns, _tt_turns_len, _every_turn_len, _tt_turns_len_r, _every_turn_len_r, _response, _response_len, _label = build_one_batch(data, batch_index, conf, turn_cut_type, term_cut_type)
    ans = {'turns': _turns,
            'tt_turns_len': _tt_turns_len,
            'every_turn_len': _every_turn_len,
            'tt_turns_len_r': _tt_turns_len_r,
            'every_turn_len_r': _every_turn_len_r,
            'response': _response,
            'response_len': _response_len,
            'label': _label}
    return ans
    

def build_batches(data, conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns_batches1 = []
    _turns_batches2 = []
    _tt_turns_len_batches1 = []
    _every_turn_len_batches1 = []
    _tt_turns_len_batches2 = []
    _every_turn_len_batches2 = []

    _response_batches = []
    _response_len_batches = []

    _label_batches = []

    batch_len = int(len(data['y'])/conf['batch_size'])
    for batch_index in range(batch_len):
        _turns1, _turns2, _tt_turns_len1, _every_turn_len1, _tt_turns_len2, _every_turn_len2, _response, _response_len, _label = build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail')

        _turns_batches1.append(_turns1)
        _turns_batches2.append(_turns2)
        _tt_turns_len_batches1.append(_tt_turns_len1)
        _every_turn_len_batches1.append(_every_turn_len1)
        _tt_turns_len_batches2.append(_tt_turns_len2)
        _every_turn_len_batches2.append(_every_turn_len2)

        _response_batches.append(_response)
        _response_len_batches.append(_response_len)

        _label_batches.append(_label)

    ans = { 
        "turns1": _turns_batches1, "turns2": _turns_batches2, "tt_turns_len1": _tt_turns_len_batches1, "every_turn_len1":_every_turn_len_batches1,
        "tt_turns_len2": _tt_turns_len_batches2, "every_turn_len2":_every_turn_len_batches2,
        "response": _response_batches, "response_len": _response_len_batches, "label": _label_batches
    }   

    return ans 

if __name__ == '__main__':
    conf = { 
        "batch_size": 5,
        "max_turn_num": 6, 
        "max_turn_len": 20, 
        "_EOS_": 1,
    }
    train, val, test, test_human = pickle.load(open('../data_ali/data.cc.cc.pkl', 'rb'))
    print('load data success')
    
    train_batches = build_batches(train, conf)
    val_batches = build_batches(val, conf)
    test_batches = build_batches(test, conf)
    test_batches = build_batches(test_human, conf)
    print('build batches success')
    
    #pickle.dump([train_batches, val_batches, test_batches], open('../data/batches.pkl', 'wb'))
    #print('dump success')
