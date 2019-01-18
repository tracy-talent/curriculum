package LDA;

public class TopicWordPhi implements Comparable<TopicWordPhi>{
	public int wordid;
	public double pval;
	
	public int compareTo(TopicWordPhi o){
		if(this.pval < o.pval) return 1;
		else if(this.pval == o.pval) return 0;
		else return -1;
	}
}
