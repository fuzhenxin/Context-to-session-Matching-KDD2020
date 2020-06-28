import sys
import numpy as np
from sklearn.metrics import average_precision_score
from utils.get_final_eval import get_final_eval

def mean_average_precision(sort_data):
    #to do
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index+1)
    if count_1==0:
        return 0
    return sum_precision / count_1

def mean_reciprocal_rank(sort_data):
    sort_lable = [s_d[1] for s_d in sort_data]
    if 1 not in sort_lable:
        return 0
    assert 1 in sort_lable
    return 1.0 / (1 + sort_lable.index(1))

def precision_at_position_1(sort_data):
    if sort_data[0][1] == 1:
        return 1
    else:
        return 0

def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[1] for s_d in sort_data]
    select_lable = sort_lable[:k]
    if 1 not in sort_lable:
        return 0
    return 1.0 * select_lable.count(1) / sort_lable.count(1)

def evaluation_one_session(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_5

def evaluate(file_path):
    sum_m_a_p = 0
    sum_m_r_r = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0

    i = 0
    total_num = 0
    with open(file_path, 'r') as infile:
        f_w = open(file_path+".sig", "w")
        for line in infile:
            if i % 10 == 0:
                data = []
            
            tokens = line.strip().split('\t')
            data.append((float(tokens[0]), int(tokens[1])))

            if i % 10 == 9:
                #if False:
                if 1 not in [jj[1] for jj in data]:
                    pass
                else:
                    total_num += 1
                    m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
                    f_w.write(" ".join(map(str, [m_a_p, m_r_r, p_1, r_1, r_2, r_5]))+"\n")
                    sum_m_a_p += m_a_p
                    sum_m_r_r += m_r_r
                    sum_p_1 += p_1
                    sum_r_1 += r_1
                    sum_r_2 += r_2
                    sum_r_5 += r_5

            i += 1

    print('total num: %s' %total_num)

    return (1.0*sum_m_a_p/total_num, 1.0*sum_m_r_r/total_num, 1.0*sum_p_1/total_num, 
            1.0*sum_r_1/total_num, 1.0*sum_r_2/total_num, 1.0*sum_r_5/total_num)


def evaluate_human(file_name, data_dir):
    get_final_eval(file_name, data_dir)
    result = evaluate(file_name+".to_score.txt")
    return result

if __name__ == '__main__':
    get_final_eval(sys.argv[1])
    result = evaluate(sys.argv[1]+".to_score.txt")
    print("MAP: {:01.4f} MRR: {:01.4f} P@1 {:01.4f} R@1 {:01.4f} r@2 {:01.4f} r@5 {:01.4f}".format(*result))
