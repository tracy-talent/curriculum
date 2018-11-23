package main.java.SortedDocInvertedIndex;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;

public class SortedDocInvertedIndexMapper extends Mapper<Object, Text, Text, IntWritable> {
    private Set<String> stopwords;
    private URI[] cacheFiles;
    public final static IntWritable one = new IntWritable(1);

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
        stopwords = new TreeSet<String>();
        Configuration conf = context.getConfiguration();
        FileSystem fs = FileSystem.get(conf);
        cacheFiles = context.getCacheFiles();
        for (int i = 0; i < cacheFiles.length; i++) {
            String line;
            Path path = new Path(cacheFiles[i]);
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));
            while ((line = reader.readLine()) != null) {
                StringTokenizer itr = new StringTokenizer(line);
                while (itr.hasMoreTokens()) {
                    stopwords.add(itr.nextToken());
                }
            }
            reader.close();
        }
    }

    @Override
    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {
        FileSplit fileSplit = (FileSplit)context.getInputSplit();
        String fileName = fileSplit.getPath().getName();
        int spos = fileName.indexOf(".txt.segmented");
        if (spos == -1) {
            spos = fileName.indexOf(".TXT.segmented");
        }
        fileName = fileName.substring(0, spos);
        String temp;
        String line = value.toString().toLowerCase();
        StringTokenizer itr = new StringTokenizer(line);
        while (itr.hasMoreTokens()) {
            temp = itr.nextToken();
            if (!stopwords.contains(temp)) {
                Text word = new Text();
                word.set(temp + "#" + fileName);
                context.write(word, one);
            }
        }
    }
}
