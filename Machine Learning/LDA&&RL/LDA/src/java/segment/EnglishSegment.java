package segment;

import java.util.*;

import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;


public class EnglishSegment extends Segment{
    //词性限制
    public Set<String> expectedTag = new HashSet<String>() {{
		/*add("n");add("v");add("vd");add("vn");add("vf");
        add("vx");add("vi");add("vl");add("vg");add("ns");
        add("nt");add("nz");add("nw");add("nl");add("nr");
        add("ng");add("userDefine");add("a");add("ad");
        add("an");add("ag");add("al");*/
        add("VB");add("VBD");add("VBG");add("VBN");add("VBP");add("VBZ");
        add("NN");add("NNS");add("NNP");add("NNPS");
        add("JJ");add("JJR");add("JJS");
    }};

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

    public String segment(String text){
    	StringBuilder sb = new StringBuilder();

        Annotation document = new Annotation(text);    // 利用text创建一个空的Annotation
        pipeline.annotate(document);                   // 对text执行所有的Annotators（七种）

        // 下面的sentences 中包含了所有分析结果，遍历即可获知结果。
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
//        System.out.println("word\tpos");

        for(CoreMap sentence: sentences) {
            for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                String word = token.get(TextAnnotation.class);            // 获取分词
                String pos = token.get(PartOfSpeechAnnotation.class);     // 获取词性标注
                if (word.replaceAll("[^a-zA-Z]", "").length() == word.length()) {
                    if (expectedTag.contains(pos) && !stopwords.contains(word)) {
                        String lemma = token.get(LemmaAnnotation.class);
                        sb.append(lemma + " ");
//                        System.out.println(word + "\t" + lemma + "\t" + pos);
                    }
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