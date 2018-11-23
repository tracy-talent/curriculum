package main.java.SortedDocInvertedIndex;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.net.URI;
import java.util.Random;

public class SortedDocInvertedIndexMain {
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 3) {
            System.err.println("Usage: hadoop jar SortedDocInvertedIndex.jar SortedDocInvertedIndexMain input soutput");
            System.exit(3);
        }
        Job job = Job.getInstance(conf, "DocInvertedIndex");
        Path tempdir = new Path("hdfs://192.168.1.1:9000/user/2018st32/SortedDocInvertedIndex-temp-"
                + Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
        job.addCacheFile(new URI("hdfs://192.168.1.1:9000/user/2018st32/stopwords.txt"));
        job.setJarByClass(SortedDocInvertedIndexMain.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        job.setPartitionerClass(SortedDocInvertedIndexPartitioner.class);
        job.setCombinerClass(SortedDocInvertedIndexCombiner.class);
        job.setMapperClass(SortedDocInvertedIndexMapper.class);
        job.setReducerClass(SortedDocInvertedIndexReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        job.setNumReduceTasks(6);
        FileInputFormat.addInputPath(job, new Path(otherArgs[1]));
        FileOutputFormat.setOutputPath(job, tempdir);
        if (job.waitForCompletion(true)) {
            Job sortedjob = Job.getInstance(conf, "SortedDocInvertedIndex");
            sortedjob.setJarByClass(SortedDocInvertedIndexMain.class);
            sortedjob.setInputFormatClass(SequenceFileInputFormat.class);
            sortedjob.setOutputFormatClass(TextOutputFormat.class);
            sortedjob.setMapOutputKeyClass(DoubleWritable.class);
            sortedjob.setMapOutputValueClass(Text.class);
            sortedjob.setMapperClass(InverseMapper.class);
            sortedjob.setNumReduceTasks(1);
            sortedjob.setOutputKeyClass(DoubleWritable.class);
            sortedjob.setOutputValueClass(Text.class);
            sortedjob.setSortComparatorClass(IntWritableDecreasingComparator.class);
            FileInputFormat.addInputPath(sortedjob, tempdir);
            FileOutputFormat.setOutputPath(sortedjob, new Path(otherArgs[2]));
            Boolean flag = sortedjob.waitForCompletion(true);
            FileSystem.get(conf).delete(tempdir, true);
            System.exit(flag ? 0 : 1);
        }
    }
}
