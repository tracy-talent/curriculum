import PCA
import SVD
import ISOMAP

if __name__ == '__main__':
    resfile = open('../result/result.csv', 'w')
    resfile.write('algorithm, dataset, K, accuracy\n')
    PCA.main(resfile)
    SVD.main(resfile)
    ISOMAP.main(resfile)
    resfile.close()