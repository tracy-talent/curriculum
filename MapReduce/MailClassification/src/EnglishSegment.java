import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.util.List;
import java.util.Properties;


public class EnglishSegment extends Segment {
    StanfordCoreNLP pipeline;

    public EnglishSegment(){
        super();
        initialConfig();
    }

    public EnglishSegment(String stopwordPath){
        super(stopwordPath);//设置停用词
        initialConfig();
    }

    public void initialConfig() {
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");    // 七种Annotators
        pipeline = new StanfordCoreNLP(props);    // 依次处理
    }

    public String segment(String text) {
    	StringBuilder sb = new StringBuilder();

        Annotation document = new Annotation(text);    // 利用text创建一个空的Annotation
        pipeline.annotate(document);                   // 对text执行所有的Annotators（七种）

        // 下面的sentences 中包含了所有分析结果，遍历即可获知结果。
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);

        for(CoreMap sentence: sentences) {
            for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                String word = token.get(TextAnnotation.class);            // 获取分词
                    if (!stopwords.contains(word)) {
                        String lemma = token.get(LemmaAnnotation.class);
                        sb.append(lemma + " ");
                    }
            }
        }
        return sb.toString().trim();
    }

    public static void main(String[] args){
    	String content = "The NASA September-October term jury had been charged by Fulton Superior Court Judge Durwood Pye,to investigate reports of possible irregularities in the \"hard-fought\",primary which was. won by";
    	//String content="Clusters are created from 433short 4.5 snippets。9 of documents retrieved by web search engines which are as good as clusters created from the full text of web documents.";
        EnglishSegment es=new EnglishSegment();
        String result=es.segment(content);
        System.out.println(result);
    }
}