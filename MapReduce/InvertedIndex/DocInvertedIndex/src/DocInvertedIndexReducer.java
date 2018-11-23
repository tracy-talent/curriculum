package main.java.DocInvertedIndex;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DocInvertedIndexReducer extends Reducer<Text, IntWritable, Text, Text> {
    private Text word1 = new Text();
    private Text word2 = new Text();
    private Text currentWord = new Text("");
    private List<String> postingList = new ArrayList<String>();

    /**
     * 计算每个词的评剧出现次数并写入结果
     * @param key
     * @param values
     * @param context
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
        int sum = 0;
        word1.set(key.toString().split("#")[0]);  //分割获取词
        String bookname = key.toString().split("#")[1];   //分割获取书名
        for (IntWritable val: values) {
            //对词在当前文档中的出现次数进行累加
            sum += val.get();
        }
        word2.set(bookname + ":" + sum);

        if (!currentWord.equals(word1) && currentWord.getLength() != 0) {
            StringBuilder out = new StringBuilder();
            long count = 0;
            //对词在不同文档中的出现次数进行累加
            for (String item: postingList) {
                out.append(item);
                out.append(";");
                count += Long.parseLong(item.substring(item.indexOf(":") + 1));
            }
            //格式化平均出现次数保留两位小数，并将词在每个文档中的出现频次增添到value中
            StringBuilder out1 = new StringBuilder(String.format("%.2f", (double)count/postingList.size()) + ",")
                    .append(out);
            if (count > 0) {
                //去除行尾的分号，然后写入reduce的键值对结果<Text,Text>
                context.write(currentWord, new Text(out1.substring(0, out1.lastIndexOf(";"))));
            }
            postingList = new ArrayList<String>();
        }
        //attention:currentWord=word1是把引用赋值给currentWord，word1变化会导致currentWord变化，所以要用new Text
        currentWord = new Text(word1);
        postingList.add(word2.toString());
    }

    /**
     * 计算当前reducer中最后一个词的平均出现次数并写入结果
     * @param context
     * @throws IOException
     * @throws InterruptedException
     */
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
        StringBuilder out1 = new StringBuilder(String.format("%.2f", (double)count/postingList.size()) + ",")
                .append(out);
        if (count > 0) {
            context.write(currentWord, new Text(out1.substring(0, out1.lastIndexOf(";"))));
        }
    }
}
