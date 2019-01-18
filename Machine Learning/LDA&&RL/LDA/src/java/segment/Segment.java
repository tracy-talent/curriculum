package segment;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * 分词抽象类
 * @author Angela
 */
public abstract class Segment {

    protected Set<String> stopwords;//停用词

    /**
     * 构造函数，初始化各个属性
     */
    public Segment(){
        stopwords=new HashSet<String>();
    }

    /**
     * 构造函数，初始化各个属性，初始化停用词集
     * @param stopwordPath 停用词文件路径
     */
    public Segment(String stopwordPath) {
    	if(!stopwordPath.isEmpty()){
    		stopwords=new HashSet<String>();
    		try{
    			BufferedReader reader = new BufferedReader(new FileReader(stopwordPath));
    			String sw = null;
    			while((sw = reader.readLine()) != null){
    				stopwords.add(sw);
    			}
    			reader.close();
    		}catch(Exception e){
    			e.printStackTrace();
    		}
    	}
        
    }

    /**
     * 对字符串内容进行分词
     * @param content 内容
     * @return 由空格符作为分隔符的分词结果String
     */
    public abstract String segment(String content);

    /**
     * @return the stopwords
     */
    public Set<String> getStopwords() {
        return stopwords;
    }

    /**
     * @param stopwords the stopwords to set
     */
    public void setStopwords(Set<String> stopwords) {
        this.stopwords = stopwords;
    }

}