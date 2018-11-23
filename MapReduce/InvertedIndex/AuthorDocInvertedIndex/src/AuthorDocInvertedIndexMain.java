package main.java.AuthorDocInvertedIndex;

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

public class AuthorDocInvertedIndexMain {
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        String[] otherArgs = new  GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 3) {
            System.out.println("Usage: hadoop jar AuthorDocInvertedIndex.jar AuthorDocInvertedIndexMain input aoutput");
            System.exit(3);
        }
        Job job = Job.getInstance(conf, "AuthorDocInvertedIndex");
        job.addCacheFile(new URI("hdfs://192.168.1.1:9000/user/2018st32/stopwords.txt"));
        job.setJarByClass(AuthorDocInvertedIndexMain.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setPartitionerClass(AuthorDocInvertedIndexPartitioner.class);
        job.setCombinerClass(main.java.DocInvertedIndex.DocInvertedIndexCombiner.class);
        job.setMapperClass(AuthorDocInvertedIndexMapper.class);
        job.setReducerClass(AuthorDocInvertedIndexReducer.class);
        job.setNumReduceTasks(6);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(otherArgs[1]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
