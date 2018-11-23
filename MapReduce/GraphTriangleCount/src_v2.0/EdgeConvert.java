package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.net.URI;
import java.util.*;

/**
 * convert directed edge to undirected edge
 */
public class EdgeConvert {

    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        conf.set("mapreduce.reduce.memory.mb", "2048");  //设置reduce container的内存大小
        conf.set("mapreduce.reduce.java.opts", "-Xmx2048m");  //设置reduce任务的JVM参数
//        conf.set("mapreduce.map.java.opts", "-Xmx2048m");  //设置map任务的JVM参数

        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: hadoop jar EdgeConvert.jar EdgeConvert " +
                    "input/graphTriangleCount/twitter_graph_v2.txt temp/EdgeConvert");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "EdgeConvert");
        job.setJarByClass(EdgeConvert.class);
        job.setMapperClass(EdgeConvertMapper.class);
        job.setReducerClass(EdgeConvertReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);
        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        SequenceFileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        SequenceFileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        job.setNumReduceTasks(TriangleCountDriver.ReducerNum);
        if(!job.waitForCompletion(true)) {
            System.exit(1);
        }
    }

    public static class EdgeConvertMapper extends Mapper<IntWritable, IntWritable, IntWritable, IntWritable> {
        @Override
        public void map(IntWritable key, IntWritable value, Context context)
                throws IOException, InterruptedException {
            if (key.get() != value.get()) {
                context.write(key, value);
                context.write(value, key);
            }
        }
    }

    public static class EdgeConvertReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
        private final static String outDegreePath = TriangleCountDriver.HDFS_PATH +
                TriangleCountDriver.OutDegreeStatPath + "part-r-00000";

        private Map<Integer, Integer> degree;
        private Map<Integer, Boolean> edgeExisted;

        @Override
        public void setup(Context context)
                throws IOException, InterruptedException {
            degree = new HashMap<Integer, Integer>();

            Configuration conf = context.getConfiguration();
            SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(new Path(outDegreePath)));
//            Writable key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), fs.getConf());
//            Writable value = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), fs.getConf());
            IntWritable key = new IntWritable();
            IntWritable value = new IntWritable();
            int cnt = 0;
            while (reader.next(key, value)) {
                degree.put(value.get(), cnt);
                cnt++;
            }
            reader.close();

        }

        @Override
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException{
            IntWritable vertex = new IntWritable();
            List<Integer> outvertex = new ArrayList<Integer>();
            edgeExisted = new HashMap<Integer, Boolean>();
            for (IntWritable val: values) {
                if (!edgeExisted.containsKey(val.get())) {
                    edgeExisted.put(val.get(), true);
                    if (degree.get(val.get()) > degree.get(key.get())) {
                        outvertex.add(val.get());
                    }
                }
            }
            Collections.sort(outvertex);
            for (Integer vt: outvertex) {
                vertex.set(vt);
                context.write(key, vertex);
            }
        }
    }

    public static class ComparatorString implements Comparator<String> {
        public int compare(String a, String b) {
            if (a.length() > b.length()) {
                return 1;
            } else if (a.length() < b.length()){
                return -1;
            } else {
                return a.compareTo(b);
            }
        }
    }
}
