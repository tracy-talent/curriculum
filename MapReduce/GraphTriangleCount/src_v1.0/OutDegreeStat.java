package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.util.Random;
import java.util.StringTokenizer;


/**
 * count the degree of every vertex
 */

public class OutDegreeStat {
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        conf.set("mapreduce.reduce.memory.mb", "2048");  //设置reduce container的内存大小
        conf.set("mapreduce.reduce.java.opts", "-Xmx2048m");  //设置reduce任务的JVM参数
//        conf.set("mapreduce.map.java.opts", "-Xmx2048m");  //设置map任务的JVM参数

        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 2) {
            System.err.println("Usage: hadoop jar OutDegreeStat.jar OutDegreeStat " +
                    "input/graphTriangleCount/twitter_graph_v2.txt temp/OutDegreeStat");
            System.exit(2);
        }
        Path tempdir = new Path(TriangleCountDriver.HDFS_PATH + "OutDegreeStat-temp-"
                + Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
        Job job = Job.getInstance(conf, "OutDegreeStat");
        job.setJarByClass(OutDegreeStat.class);
        job.setMapperClass(OutDegreeStatMapper.class);
        job.setReducerClass(OutDegreeStatReducer.class);
        job.setCombinerClass(OutDegreeStatCombiner.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        SequenceFileOutputFormat.setOutputPath(job, tempdir);
        job.setNumReduceTasks(TriangleCountDriver.ReducerNum);
        if (job.waitForCompletion(true)) {
            Job sortedJob = Job.getInstance(conf, "SortedOutDegree");
            sortedJob.setJarByClass(OutDegreeStat.class);
            sortedJob.setMapperClass(InverseMapper.class);
            sortedJob.setMapOutputKeyClass(IntWritable.class);
            sortedJob.setMapOutputValueClass(Text.class);
            sortedJob.setOutputKeyClass(IntWritable.class);
            sortedJob.setOutputValueClass(Text.class);
            sortedJob.setInputFormatClass(SequenceFileInputFormat.class);
            sortedJob.setOutputFormatClass(SequenceFileOutputFormat.class);
            SequenceFileInputFormat.addInputPath(sortedJob, tempdir);
            SequenceFileOutputFormat.setOutputPath(sortedJob, new Path(otherArgs[1]));
            sortedJob.setNumReduceTasks(1);
            boolean flag = sortedJob.waitForCompletion(true);
            FileSystem.get(conf).delete(tempdir, true);
            if (!flag) {
                System.exit(1);
            }
        }
    }

    public static class OutDegreeStatMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final IntWritable one = new IntWritable(1);

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer itr = new StringTokenizer(line);
            Text vertex1 = new Text(itr.nextToken());
            Text vertex2 = new Text(itr.nextToken());
            if (!vertex1.equals(vertex2)) {
                context.write(vertex1, one);
                context.write(vertex2, one);
            }
        }
    }

    public static class OutDegreeStatReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val: values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    public static class OutDegreeStatCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val: values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }
}
