


def merge_human():
    files = ["human.res0.crowd", "human.res1.crowd", "human.res2.crowd"]
    scores = []
    for file_name in files:
        lines = open(file_name, "r").readlines()
        lines = [int(i.strip()) for i in lines]
        for i in lines:
            assert i in [0, 1]
        scores.append(lines)
    
    res = []
    for i in range(len(scores[0])):
        scores_per = [ j[i] for j in scores]
        scores_per_sum = sum(scores_per)
        if scores_per_sum>=2:
            res.append(1)
        else:
            res.append(0)
    f_w = open("human.res.crowd", "w")
    for i in res:
        f_w.write(str(i)+"\n")


if __name__=="__main__":
    merge_human()