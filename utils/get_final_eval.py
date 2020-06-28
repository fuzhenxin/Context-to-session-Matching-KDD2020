
import random

def get_final_eval(file_name, data_dir):
    #f_res = open("../../output_ali/cc.cc.3.con2.pos.mean.weight/score.test", "r")
    #f_res = open("../../output_ali/cc.cr.dam/score.test", "r")
    f_res = open(file_name, "r")
    lines_res = f_res.readlines()
    lines_res = [i.strip().split()[0] for i in lines_res]
    #lines_res = [random.random() for i in lines_res]
    #lines_res = [10-i%10 for i,j in enumerate(lines_res)]
    f_human = open(data_dir+"/human/human.res.crowd", "r")
    lines_human = f_human.readlines()
    lines_human = [int(i.strip()) for i in lines_human]
    f_w = open(file_name+".to_score.txt", "w")
    for i in range(10000):
        f_w.write(str(lines_res[i])+"\t"+str(lines_human[i])+"\n")

if __name__=="__main__":
    get_final_eval("../../output_ali/cc.cc.3.con2.pos.mean.weight/score.test")
