package LDA;

import segment.EnglishSegment;
import java.io.*;
import java.util.*;

public class LDA_ES {
	public static final double alpha=0.5,beta=0.1;
//	public static final int ndocs=8883,iters=200,nwords=88000,maxdoclen=3800;  //区分大小写
    public static final int ndocs=8883,iters=200,nwords=103390,maxdoclen=3855;  //不区分大小写
    public static final int B = 1000000007,mod=10000019;
	public int ntopics;
	public String OUTDIR;
    public DocWordTopic z[][] = new DocWordTopic[ndocs+1][maxdoclen];
	public int nw[][];
	public int nwsum[];
	public int nd[][];
	public DocTopicTheta theta[][];
	public TopicWordPhi phi[][];
	public DocTopicTheta thetadec[][];
	public TopicWordPhi phidec[][];
	public int ndsum[] = new int[ndocs+1];
	public int hash2index[] = new int[mod+1];
	public int wordfreq[] = new int[nwords+1];
	public String voca[] = new String[nwords+1];
	public int wordcount = 0,doccount = 0; //词计数，文档计数

    public LDA_ES(int ntopics) {
        this.ntopics = ntopics;
        OUTDIR =  "output/LDA/topic" + ntopics + "/";
        nw = new int[nwords+1][ntopics+1];
        nwsum = new int[ntopics+1];
        nd = new int[ndocs+1][ntopics+1];
        theta = new DocTopicTheta[ndocs+1][ntopics+1];
        phi = new TopicWordPhi[ntopics+1][nwords+1];
        thetadec = new DocTopicTheta[ndocs+1][ntopics+1];
        phidec = new TopicWordPhi[ntopics+1][nwords+1];
    }

	//字符串哈希值
	public int wordhash(String word){
		int len = word.length();
		long hval = 0;
		for(int i = 0;i < len;i++)
			hval = (hval*B + word.charAt(i) - 'A')%mod;
		return (int)hval;
	}
	
	//对文本进行预处理
	public void initialText(){
		wordcount = doccount = 0;
		String line = null;
		BufferedReader reader;
		Random rand = new Random();
		EnglishSegment es = new EnglishSegment("stopwords_en.txt");
		int wordnum = 0;
		try{
			reader = new BufferedReader(new FileReader("corpus/news.txt"));
			while((line=reader.readLine()) != null) {
                String segres = es.segment(line);
                if (segres.length() == 0) continue;
                doccount++;
                String term[] = segres.split(" ");
				for(int i = 0; i < term.length; i++) {
					wordnum++;
					z[doccount][wordnum] = new DocWordTopic();
					int tp = z[doccount][wordnum].topic = rand.nextInt(ntopics)+1;
                    int hval = wordhash(term[i]);
					if(hash2index[hval] == 0){
						hash2index[hval] = ++wordcount;
						voca[wordcount] = term[i];
					}
					wordfreq[hash2index[hval]]++;
					z[doccount][wordnum].wordid = hash2index[hval];
					nw[hash2index[hval]][tp]++;
					nwsum[tp]++;
					nd[doccount][tp]++;
				}
                ndsum[doccount] = wordnum;
				wordnum = 0;
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//累积概率的随机抽样
	public int cumulative(double pnum[]){
		int idx = 0;
		for(int i = 2;i < pnum.length;i++)
			pnum[i] += pnum[i-1];
		double pshield = Math.random()*pnum[pnum.length-1];
		for(int i = 1;i < pnum.length;i++)
			if(pnum[i] > pshield){
				idx = i;
				break;
			}
		return idx;
	}
	
	//Collapsed Gibbs Sampling
	//时间复杂度:迭代次数*总词数*主题数
	//空间复杂度：主题数*词数+文档数*主题数(约等于主题数*词数)
	public void CGS(){
		double p[] = new double[ntopics+1];
		for(int k = 1;k <= iters;k++) {
            for (int i = 1; i <= ndocs; i++) {
                for (int j = 1; j <= ndsum[i]; j++) {
                    int t1 = z[i][j].topic;
                    nw[z[i][j].wordid][t1]--;
                    nwsum[t1]--;
                    nd[i][t1]--;
                    for (int t = 1; t <= ntopics; t++)
                        p[t] = ((nw[z[i][j].wordid][t] + beta) / (nwsum[t] + wordcount * beta)) * ((nd[i][t] + alpha) / (ndsum[i] + ntopics * alpha));
                    int t2 = cumulative(p);
                    z[i][j].topic = t2;
                    nw[z[i][j].wordid][t2]++;
                    nwsum[t2]++;
                    nd[i][t2]++;
                }
            }
            System.out.println("第" + k + "轮迭代结束");
        }
		for(int i = 1;i <= ndocs;i++)
			for(int j = 1;j <= ntopics;j++){
				theta[i][j] = new DocTopicTheta();
				theta[i][j].tval = (nd[i][j] + alpha)/(ndsum[i] + ntopics*alpha);
				theta[i][j].topicid = j;
				thetadec[i][j] = new DocTopicTheta();
				thetadec[i][j].tval = theta[i][j].tval;
				thetadec[i][j].topicid = j;
			}
		for(int i = 1;i <= ntopics;i++)
			for(int j = 1;j <= wordcount;j++){
				phi[i][j] = new TopicWordPhi();
				phi[i][j].pval = (nw[j][i] + beta)/(nwsum[i] + wordcount*beta);
				phi[i][j].wordid = j;
				phidec[i][j] = new TopicWordPhi();
				phidec[i][j].pval = phi[i][j].pval;
				phidec[i][j].wordid = j;
			}
		for(int i = 1;i <= ndocs;i++)
			Arrays.sort(thetadec[i],1,ntopics+1);
		for(int i = 1;i <= ntopics;i++)
			Arrays.sort(phidec[i],1,wordcount+1);
	}
	
	//输出结果到文件中
	public void output(){
		BufferedWriter writer;
		try{
		    File outdir = new File(OUTDIR);
		    if (!outdir.exists()) outdir.mkdirs();
		    WordsizeId ws[] = new WordsizeId[10];
			writer = new BufferedWriter(new FileWriter(OUTDIR + "topic-top10keywords.txt"));
			writer.write("\\begin{tabular}{c p{1.8cm}<{\\centering} p{1.6cm}<{\\centering} p{1.4cm}<{\\centering} p{1.4cm}<{\\centering} p{1.2cm}<{\\centering} p{1.2cm}<{\\centering} p{1.0cm}<{\\centering} p{1.0cm}<{\\centering} p{0.8cm}<{\\centering} p{0.8cm}<{\\centering}}\n");
			writer.write("\\toprule\n&$w1$&$w2$&$w3$&$w4$&$w5$&$w6$&$w7$&$w8$&$w9$&$w10$\\\\\n");
			for(int i = 1;i <= ntopics;i++) {
			    StringBuilder sb = new StringBuilder();
				writer.write("\\hline\n\\multirow{2}*{\\shortstack{t" + i + "}}");
				for(int j = 1;j <= 10;j++)
				    ws[j-1] = new WordsizeId(phidec[i][j].wordid, voca[phidec[i][j].wordid].length(), phidec[i][j].pval);
                Arrays.sort(ws, 0, 10);
                for (int j = 0; j < 10; j++) {
                    writer.write("&" + voca[ws[j].wordid]);
                }
                writer.write("\\\\\n");
                for (int j = 0; j < 10; j++) {
                    writer.write("&" + String.format("%.2f", ws[j].pv*100) + "\\%");
                }
                writer.write("\\\\\n");
			}
			writer.write("\\bottomrule\n");
			writer.write("\\end{tabular}");
			writer.close();
		}catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void solve(){
		initialText();
		CGS();
		output();
	}
}
