import random





if __name__ == "__main__":
    f = open("./tmp/results.txt","r")
    f1 = open("./res_ckpt/id.txt","w",encoding="utf-8")
    f3 = open("./data/test_v3.txt",'r',encoding="utf-8")
    for (line,line1) in zip(f,f3):
        res = line.split("	")
        if float(res[0]) > 0.985 :
            f1.write(line1.split("\t")[0] + "\t" + str(0) + "\n")
        else:
            f1.write(line1.split("\t")[0] + "\t" + str(1) + "\n")
