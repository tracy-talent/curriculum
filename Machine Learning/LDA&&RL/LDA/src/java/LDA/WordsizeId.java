package LDA;

public class WordsizeId implements Comparable<WordsizeId>{
    public int wordlen;
    public int wordid;
    public double pv;

    public WordsizeId(int wordid, int wordlen, double pv) {
        this.wordid = wordid;
        this.wordlen = wordlen;
        this.pv = pv;
    }

    @Override
    public int compareTo(WordsizeId ws) {
        if (this.wordlen < ws.wordlen) return 1;
        else if(this.wordlen == ws.wordlen) return 0;
        else return -1;
    }
}
