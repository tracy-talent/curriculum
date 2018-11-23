package main.java.SortedDocInvertedIndex;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SortedDocInvertedIndexReducer extends Reducer<Text, IntWritable, Text, DoubleWritable> {
    private Text word1 = new Text();
    private Text currentWord = new Text("");
    private List<Integer> postingList = new ArrayList<Integer>();

    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        int sum = 0;
        word1.set(key.toString().split("#")[0]);
        for (IntWritable val: values) {
            sum += val.get();
        }

        if (!currentWord.equals(word1) && currentWord.getLength() != 0) {
            int count = 0;
            for (Integer item: postingList) {
                count += item;
            }
            if (count > 0) {
                context.write(currentWord, new DoubleWritable((double)Math.round((double)count/postingList.size()*100)/100));
            }
            postingList = new ArrayList<Integer>();
        }
        //attention:currentWord=word1是把引用赋值给currentWord，word1变化会导致currentWord变化，所以要用new Text
        currentWord = new Text(word1);
        postingList.add(sum);
    }

    @Override
    public void cleanup(Context context)
            throws IOException, InterruptedException {
        int count = 0;
        for (Integer item: postingList) {
            count += item;
        }
        if (count > 0) {
            context.write(currentWord, new DoubleWritable((double)Math.round((double)count/postingList.size()*100)/100));
        }
    }
}
