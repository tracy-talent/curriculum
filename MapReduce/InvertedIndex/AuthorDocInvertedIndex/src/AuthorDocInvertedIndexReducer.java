package main.java.AuthorDocInvertedIndex;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

public class AuthorDocInvertedIndexReducer extends Reducer<Text, IntWritable, Text, Text> {
    private Text word1 = new Text();
    private Text word2 = new Text();
    private Text currentWord = new Text("");
    private List<String> postingList = new ArrayList<String>();
    private int fileNum;

    @Override
    public void setup(Context context)
            throws IOException, InterruptedException{
        FileSystem fs = FileSystem.get(context.getConfiguration());
        fileNum = fs.listStatus(new Path("hdfs://192.168.1.1:9000/data/wuxia_novels")).length;
    }

    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        int sum = 0;
        word1.set(key.toString().split("#")[0]);
        String bookname = key.toString().split("#")[1];
        for (IntWritable val: values) {
            sum += val.get();
        }
        word2.set(bookname + ":" + sum);

        if (!currentWord.equals(word1) && currentWord.getLength() != 0) {
            StringBuilder out = new StringBuilder();
            long count = 0;
            for (String item: postingList) {
                out.append(item);
                out.append(";");
                count += Long.parseLong(item.substring(item.indexOf(":") + 1));
            }
            double tf_idf = count*Math.log(1.0*fileNum/(postingList.size() + 1));
            StringBuilder out1 = new StringBuilder(String.format("%.2f", tf_idf) + ",")
                    .append(out);
            if (count > 0) {
                context.write(currentWord, new Text(out1.substring(0, out1.lastIndexOf(";"))));
            }
            postingList = new ArrayList<String>();
        }
        //attention:currentWord=word1是把引用赋值给currentWord，word1变化会导致currentWord变化，所以要用new Text
        currentWord = new Text(word1);
        postingList.add(word2.toString());
    }

    @Override
    public void cleanup(Context context)
            throws IOException, InterruptedException {
        StringBuilder out = new StringBuilder();
        long count = 0;
        for (String item: postingList) {
            out.append(item);
            out.append(";");
            count += Long.parseLong(item.substring(item.indexOf(":") + 1));
        }
        double tf_idf = count*Math.log(1.0*fileNum/(postingList.size() + 1));
        StringBuilder out1 = new StringBuilder(String.format("%.2f", tf_idf) + ",")
                .append(out);
        if (count > 0) {
            context.write(currentWord, new Text(out1.substring(0, out1.lastIndexOf(";"))));
        }
    }
}
