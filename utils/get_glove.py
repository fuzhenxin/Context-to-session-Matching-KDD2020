
import codecs

def get_glove(file_name_in, file_name_out):
    f = codecs.open(file_name_in, "r", "utf-8")
    f_w = codecs.open(file_name_out, "w", "utf-8")
    lines = f.readlines()
    lines = [line.strip()[1:] for line in lines]
    lines = [line.replace("\t", " ").strip() for line in lines]
    for line in lines:
        f_w.write(line+"\n")

if __name__=="__main__":
    get_glove("../../../data/ecommerce/train.txt", "../../glove/train.txt")
