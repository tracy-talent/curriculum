package LDA;

public class DocTopicTheta implements Comparable<DocTopicTheta>{
	public int topicid;
	public double tval;
	
	/**从大到小进行排列*/
	public int compareTo(DocTopicTheta o){
		if(this.tval < o.tval) return 1;
		else if(this.tval == o.tval) return 0;
		else return -1;
	}
}
