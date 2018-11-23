package main.java.DocInvertedIndex;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.net.URI;

public class DocInvertedIndexMain {
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        String[] otherArgs = new  GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 3) {
            System.out.println("Usage: hadoop jar DocInvertedIndex.jar DocInvertedIndexMain input output");
            System.exit(3);
        }
        Job job = Job.getInstance(conf, "DocInvertedIndex");
        job.addCacheFile(new URI("hdfs://192.168.1.1:9000/user/2018st32/stopwords.txt"));
        job.setJarByClass(DocInvertedIndexMain.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setPartitionerClass(DocInvertedIndexPartitioner.class);
        job.setCombinerClass(DocInvertedIndexCombiner.class);
        job.setMapperClass(DocInvertedIndexMapper.class);
        job.setReducerClass(DocInvertedIndexReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(6);
        FileInputFormat.addInputPath(job, new Path(otherArgs[1]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
