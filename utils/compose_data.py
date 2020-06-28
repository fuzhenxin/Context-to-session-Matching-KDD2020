#encoding="utf-8"

import codecs
import pickle as pkl
import numpy as np

def get_voc(lines, out_file, data_type, data_prefix="", data_type_wa=""):
    voc_cnt = dict()
    voc_dict = dict()
    voc_dict["UNK"] = 0
    voc_dict["SENT"] = 1
    words_all = ["UNK", "SENT"]
    word_global_index = 2
    for line in lines:
        line = line.split("|")[:-1]
        line = "|".join(line)
        line = line.replace("|", " ")
        line = line.split()
        for word in line:
            if word in voc_cnt:
                voc_cnt[word] += 1
            else:
                voc_cnt[word] = 1
    voc_item = sorted(voc_cnt.items(), key=lambda v:(v[1]), reverse=True)
    print(voc_item[:10])
    print("Words before filt: ", len(voc_item))

    for word_per in voc_item:
        word_per = word_per[0]
        voc_dict[word_per] = word_global_index
        words_all.append(word_per)
        word_global_index += 1
    print("Words after filt: ", len(voc_item))

    f_w = codecs.open(out_file, "w", "utf-8")
    for word_per in words_all:
        f_w.write(word_per+"\n")
    write_emb(words_all, data_type, data_prefix, data_type_wa)
    return voc_dict

def write_emb(words, data_type, data_prefix, data_type_wa):
    f = codecs.open("../../glove_"+data_type_wa+"/vectors.txt", "r", "utf-8")
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    emb = dict()
    all_emb = []
    for line_index,line in enumerate(lines):
        line_w = line.split()[0]
        line_emb = np.array([float(i) for i in line.split()[1:]])
        if line_emb.shape[0]!=200:
            print("Error in: ", line_index)
            continue
        assert line_emb.shape[0]==200, (line_emb.shape[0], line_index)
        emb[line_w] = line_emb
        all_emb.append(line_emb)
    all_emb = np.array(all_emb)
    print(all_emb.shape)
    all_emb = np.mean(all_emb, axis=0)
    print("UNK emc shape: ", all_emb.shape)
    res = []
    for word in words:
        if word in emb:
            res.append(emb[word])
        else:
            res.append(all_emb)
    res = np.array(res)
    print("Glove shape: ", res.shape)
    pkl.dump(res, open("../data_"+data_type_wa+"/glove."+data_prefix+"."+data_type+".pkl", "wb"))


def convert_to_id(line, voc):
    #print("==============")
    #print(line)
    line = line.replace("\t", " | ")
    line = line.split()
    ids = []
    for word in line:
        if word == "|":
            word_id = voc["SENT"]
        else:
            word_id = voc[word] if word in voc else voc["UNK"]
        ids.append(word_id)
    return ids

def compose_data(data_type, data_prefix="", data_type_wa="", voc=None):
    res = []
    
    file_name = "../data_"+data_type_wa+"/"+data_prefix+ "." +data_type+ "."
    file_names = ["train", "valid", "test", "test.human"]
    for file_name_per in file_names:
        f = codecs.open(file_name+file_name_per, "r", "utf-8")
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        res_c1, res_c2, res_r, y = [], [], [], []
        for line in lines:
            line = line.split("|")
            assert len(line)==4
            ids_c1 = convert_to_id(line[0], voc)
            ids_c2 = convert_to_id(line[1], voc)
            ids_r = convert_to_id(line[2], voc)
            res_c1.append(ids_c1)
            res_c2.append(ids_c2)
            res_r.append(ids_r)
            y.append(int(line[-1]))
        tvt = {"c1": res_c1, "c2": res_c2, "r": res_r, "y": y}
        res.append(tvt)

    pkl.dump(res, open("../data_"+data_type_wa+"/data."+data_prefix+"."+data_type+".pkl", "wb"))

if __name__=="__main__":
    data_prefix = "cc"
    data_type_wa = "ali"
    data_type = "cr"
    file_name = "../data_"+data_type_wa+"/"+data_prefix+ "." +data_type+ "."
    f = codecs.open(file_name+"train", "r", "utf-8")
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    voc = get_voc(lines, "../data_"+data_type_wa+"/voc."+data_prefix+"."+data_type, data_type, data_prefix, data_type_wa)
    compose_data("cr", data_prefix, data_type_wa, voc)
    compose_data("cc", data_prefix, data_type_wa, voc)

