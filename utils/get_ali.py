#encoding="utf-8"

import codecs
import random
import numpy as np
import numpy.linalg as LA
from sklearn.feature_extraction.text import TfidfVectorizer
from Search import Search
import datetime
from multiprocessing import Pool

class Data():
    def __init__(self, file_name_train, file_name_test, trans=False, data_type=""):
        self.trunc_size = 20
        self.read_ali_data(file_name_train, file_name_test)
        if trans:
            self.trans_tfidf()
        self.cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)+0.0000001), 3)

        self.search = Search("../../data/ecommerce")
        self.search_error = 0
        self.data_type = data_type

    def read_ali_data(self, file_name_train, file_name_test):
        f = codecs.open(file_name_train, "r", "utf-8")
        lines = f.readlines()
        # to do by zhenxin fu
        lines = [ i.split("\t")[0]+"\t"+"\t".join(i.split("\t")[-4:]) for i in lines ]
        self.cry = [line.strip().replace("|", " ") for line in lines]
        lines_r = [line[2:] for line in self.cry if line.split("\t")[0]=="1"]
        self.lines_context = ["\t".join(line.split("\t")[:-1]) for line in lines_r]
        self.lines_last = [line.split("\t")[-2] for line in lines_r]
        self.lines_res = [line.split("\t")[-1] for line in lines_r]
        self.lines_res_sample = self.lines_res[:-2000]

        f = codecs.open(file_name_test, "r", "utf-8")
        lines_t = f.readlines()
        # to do by zhenxin fu
        lines_t = [ i.split("\t")[0]+"\t"+"\t".join(i.split("\t")[-4:]) for i in lines_t ]
        self.cry_t = [line.strip().replace("|", " ") for line in lines_t]

        rc_dict = dict()
        print("Num:", len(lines_r))
        for line in lines_r:
            line_split = line.split("\t")
            r = line_split[-1]
            c = "\t".join(line_split[:-1])
            c_index = c[-20:]
            if r in rc_dict:
                if c_index not in rc_dict[r]:
                    rc_dict[r][c_index] = c
            else:
                rc_dict[r] = {c_index:c}
        self.rc_dict = dict()
        for r,cs in rc_dict.items():
            self.rc_dict[r] = list(cs.values())

        """
        total_value = 0
        for r,c in self.rc_dict.items():
            c = c[:100]
            if len(c)<4:
                continue
            total_value += len(c)
        print("Total v", total_value)
        exit()
        """
        #self.show_case()

    def show_case(self):
        show_items = sorted(self.rc_dict.items(), key=lambda k:(len(k[1])), reverse=True)
        f = codecs.open("fre.show", "w", "utf-8")
        last_len = 12
        for index, (i,j) in enumerate(show_items):
            #print("=========")
            #print(i)
            f.write("============================\n")
            f.write(i+"\n")
            f.write(str(len(j))+"\n")
            if len(j)!=last_len:
                last_len = len(j)
                f.write("FZX"+str(len(j))+"FZX\n")
            for jj in j[:30]:
                #print(jj)
                f.write(jj+"\n")
            #if index>10:
            #    break
        #print("Finish show")

    def trans_tfidf(self):
        self.vectorizer_context = TfidfVectorizer(max_features=50000)
        self.vectorizer_context.fit_transform([i.replace("\t", " ") for i in self.lines_context]) # to do
        self.vectorizer_last = TfidfVectorizer(max_features=50000)
        self.vectorizer_last.fit_transform(self.lines_last) # to do
        
    def search_similar(self, lines, trans_type="context"):
        #print("Before trans")
        if trans_type=="context":
            lines_trans = self.vectorizer_context.transform(lines)
        elif trans_type=="last":
            lines = [i.split("\t")[-1] for i in lines]
            lines_trans = self.vectorizer_last.transform(lines)
        else:
            assert False, "trans_type error"
        #print("After trans")
        lines_trans = lines_trans.toarray()
        #print(lines_trans.shape)
        trans_cosine = [ [self.cx(ii,jj) if i!=j else -0.1 for j,jj in \
                enumerate(lines_trans)]  for i,ii in enumerate(lines_trans) ]
        return np.array(trans_cosine)

    def search_pair(self, trans_type=""):

        def process_per(rc):
            r = rc[0]
            c = rc[1]
            c = c[:100]
            c_num = len(c)
            #print(c_num)
            count_1_per, count_2_per = 0, 0
            cc_pairs_per, ccr_pairs_per = [], []
            if c_num<2:
                return 0, 0, []
            trunc_count = int(c_num/self.trunc_size)
            for trunc_index in range(trunc_count):
                if c_num-trunc_index*self.trunc_size<2: continue
                c_cur = c[trunc_index*self.trunc_size:(trunc_index+1)*self.trunc_size]

                c_cos = self.search_similar(c_cur, trans_type=trans_type)
                c_index = np.argmax(c_cos, axis=1)
                c_max = np.max(c_cos, axis=1)
                c_max_mean = np.mean(c_max)
                for c_case, c_case_index, c_case_max in zip(c_cur, c_index, c_max):
                    #if not c_case_max>c_max_mean:
                    #    continue
                    c_case_pair = c_cur[c_case_index]
                    if len(c_case_pair)<len(c_case):
                        c_case, c_case_pair = c_case_pair, c_case
                        count_1_per += 1
                    else:
                        count_2_per += 1
                    #if random.random()<0.5:
                    #    c_case, c_case_pair = c_case_pair, c_case
                    ccr_pairs_per.append(c_case+"\t\t\t"+c_case_pair+"\t\t\t"+r)
            return count_1_per, count_2_per, ccr_pairs_per

        total_count = 0
        cc_pairs = []
        ccr_pairs = []
        count_1, count_2 = 0, 0
        print("Begin similarity search", len(self.rc_dict.keys()))
        s = 0
        for i,j in self.rc_dict.items():
            s += len(j)
        print(s)
        #cc_count = 0
        in_rcs = []
        for r, c in self.rc_dict.items():
            in_rcs.append([r,c])
        #in_rcs = in_rcs[:1000] # to do
        #import pdb; pdb.set_trace()
        #with Pool(5) as p:
        #    res = p.map(process_per, in_rcs)
        if True:
            res = []
            for r, c in in_rcs:
                res.append(process_per([r,c]))
            for con_1, con_2, ccr_per in res:
                count_1 += con_1
                count_2 += con_2
                ccr_pairs.extend(ccr_per)

        print("End similarity search")
        print(count_1, count_2)
        ccr_pairs = list(set(ccr_pairs))
        print("Length ccr: {}".format(len(ccr_pairs)))

        self.ccr_pairs = [[i.split("\t\t\t")[0], i.split("\t\t\t")[1], i.split("\t\t\t")[2]] for i in ccr_pairs]
        self.contexts_negative = list(set([i[0]+"|"+i[2] for i in self.ccr_pairs]+[i[1]+"|"+i[2] for i in self.ccr_pairs]))
        #random.seed(1027)
        #random.shuffle(self.ccr_pairs)
        return
    
    def search_res(self, line, search_type=1):
        # search_type 1 context 2 last
        return_count = 12
        if search_type==2:
            line_s = line.split("\t")[-1]
        else:
            line_s = line
        line_s = "\t".join([ i[:100] for i in line_s.split("\t")[:5]])
        search_res = self.search.search_one(line_s, search_type, return_count=return_count)

        res = []
        if not search_res:
            search_res = []
        for search_per in search_res:
            if search_per[1][-20:] == line[-20:]:
                continue
            else:
                res.append(search_per)
                if len(res)==10:
                    break
        if len(res)!=10:
            print(len(res), end=" ")
            self.search_error+=1
        if len(res)<10:
            res = res*10
            res = res[:10]
        if len(res)==0:
            res = [[ "unk", "unk", "unk", "unk", 0.0 ] for i in range(10)]
        return res
        

    def write_pair_cc(self, data_prefix=""):
        random.seed(1027)

        cc_right = []
        for ccr_pair in self.ccr_pairs[:-2000]:
            cc_right.append(ccr_pair[0]+"|"+ccr_pair[1]+"|"+ccr_pair[2]+"|1\n")
            cc_right.append(ccr_pair[1]+"|"+ccr_pair[0]+"|"+ccr_pair[2]+"|1\n")
        cc_right = list(set(cc_right))
        cc_wrong = []
        for ccr_pair in self.ccr_pairs[:-2000]:
            cc_wrong.append(ccr_pair[0]+"|"+random.choice(self.contexts_negative)+"|0\n")
            cc_wrong.append(ccr_pair[1]+"|"+random.choice(self.contexts_negative)+"|0\n")
        cc_wrong = list(set(cc_wrong))

        cc_all = cc_right+cc_wrong
        random.shuffle(cc_all)
        
        f_w_name = "../data_"+self.data_type+"/"+data_prefix +".cc."
        f_w = codecs.open(f_w_name+"train", "w", "utf-8")
        for line in cc_all:
            f_w.write(line)

        f_w = codecs.open(f_w_name+"valid", "w", "utf-8")
        for ccr_pair in self.ccr_pairs[-2000:-1000]:
            f_w.write(ccr_pair[0]+"|"+ccr_pair[1]+"|"+ccr_pair[2]+"|1\n")
            for i in range(9):
                f_w.write(ccr_pair[0]+"|"+random.choice(self.contexts_negative)+"|0\n")
            f_w.write(ccr_pair[1]+"|"+ccr_pair[0]+"|"+ccr_pair[2]+"|1\n")
            for i in range(9):
                f_w.write(ccr_pair[1]+"|"+random.choice(self.contexts_negative)+"|0\n")                

        f_w = codecs.open(f_w_name+"test", "w", "utf-8")
        for ccr_pair in self.ccr_pairs[-1000:]:
            f_w.write(ccr_pair[0]+"|"+ccr_pair[1]+"|"+ccr_pair[2]+"|1\n")
            for i in range(9):
                f_w.write(ccr_pair[0]+"|"+random.choice(self.contexts_negative)+"|0\n")
            f_w.write(ccr_pair[1]+"|"+ccr_pair[0]+"|"+ccr_pair[2]+"|1\n")
            for i in range(9):
                f_w.write(ccr_pair[1]+"|"+random.choice(self.contexts_negative)+"|0\n") 

        self.filt_cr = dict()
        for i,j,k in self.ccr_pairs[-2000:]:
            self.filt_cr[i+"\t"+k] = 1
            self.filt_cr[j+"\t"+k] = 1


        
        f_w = codecs.open(f_w_name+"test.human", "w", "utf-8")
        f_w_a = codecs.open(f_w_name+"test.human.tfscore", "w", "utf-8")
        lines = [line for line in self.cry_t if line[0]=="1"]
        for line in lines:
            line_r = line.split("\t")[-1]
            line_c = "\t".join(line.split("\t")[1:-1])
            if data_prefix=="cc":
                res = self.search_res(line_c)
            elif data_prefix=="ll":
                res = self.search_res(line_c, search_type=2)
            else:
                assert False
            assert len(res)==10
            for res_per in res:
                ret_c = res_per[1]
                ret_l = res_per[2]
                ret_r = res_per[3]
                f_w.write(line_c+"|"+ret_c+"|"+ret_r+"|1\n")
                f_w_a.write(ret_r+" "+str(res_per[4]) +"\n")
        

    def filt_cr_fun(self):
        # self.cry
        res = []
        error_count = 0
        for i in self.cry:
            ii = i[2:]
            if ii in self.filt_cr:
                error_count += 1
            else:
                res.append(i)
        self.cry = res
        print(self.cry[0])
        print("Filt count: ", error_count)
        #print(self.filt_cr.keys()[0])
        
    def write_pair_cr(self, data_prefix=""):

        self.filt_cr_fun()

        f_w_name = "../data_"+self.data_type+"/"+data_prefix+".cr."
        f_w = codecs.open(f_w_name+"train", "w", "utf-8")
        for line in self.cry[:-10000]:
            line_split = line.split("\t")
            f_w.write("\t".join(line_split[1:-1])+"|UNK|"+line_split[-1]+"|"+line_split[0]+"\n")
            if self.data_type=="weibo":
                f_w.write("\t".join(line_split[1:-1])+"|UNK|"+random.choice(self.lines_res_sample)+"|0\n")

        f_w = codecs.open(f_w_name+"valid", "w", "utf-8")
        for line in self.cry[-10000:]:
            line_split = line.split("\t")
            f_w.write("\t".join(line_split[1:-1])+"|UNK|"+line_split[-1]+"|"+line_split[0]+"\n")
            if self.data_type=="weibo":
                f_w.write("\t".join(line_split[1:-1])+"|UNK|"+random.choice(self.lines_res_sample)+"|0\n")

        f_w = codecs.open(f_w_name+"test", "w", "utf-8")
        for line in self.cry_t:
            line_split = line.split("\t")
            f_w.write("\t".join(line_split[1:-1])+"|UNK|"+line_split[-1]+"|"+line_split[0]+"\n")
            if self.data_type=="weibo":
                for i in range(10):
                    f_w.write("\t".join(line_split[1:-1])+"|UNK|"+random.choice(self.lines_res_sample)+"|0\n")

        """
        f_w = codecs.open(f_w_name+"test.cr", "w", "utf-8")
        f_w_r_a = codecs.open(f_w_name+"test.r.a", "w", "utf-8")
        f_w_r = codecs.open(f_w_name+"test.r", "w", "utf-8")
        lines = [line for line in self.cry_t if line[0]=="1"]
        for line in lines:
            f_w_r.write(line.split("\t")[-1]+"\n")
            line_c = "\t".join(line.split("\t")[1:-1])
            line_last = line.split("\t")[-2]
            line_r = line.split("\t")[-1]
            if data_prefix=="cc":
                res = self.search_res(line_c)
            elif data_prefix=="ll":
                res = self.search_res(line_c, search_type=2)
            else:
                assert False
            assert len(res)==10
            for res_per in res:
                ret_c = res_per[1]
                ret_l = res_per[2]
                ret_r = res_per[3]
                f_w.write(line_c+"|"+ret_r+"|1\n")
                f_w_r_a.write(ret_r+"\n")
        """


def get_ali():
    file_name_train = "../../../data/ecommerce/train.txt"
    file_name_test = "../../../data/ecommerce/test.txt"
    data = Data(file_name_train, file_name_test, trans=True, data_type="ali")
    data.search_pair(trans_type="context")
    data_prefix = "cc"
    data.write_pair_cc(data_prefix=data_prefix)
    data.write_pair_cr(data_prefix=data_prefix)
    print(data.search_error)

if __name__=="__main__":
    get_ali()

