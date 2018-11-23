package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.util.*;

public class GraphTriangleCount {

    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        conf.set("mapreduce.reduce.memory.mb", "2048");  //设置reduce container的内存大小
        conf.set("mapreduce.reduce.java.opts", "-Xmx2048m");  //设置reduce任务的JVM参数
//        conf.set("mapreduce.map.java.opts", "-Xmx2048m");  //设置map任务的JVM参数

        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 2) {
            System.err.println("Usage: hadoop jar GraphTriangleCount.jar GraphTriangleCount " +
                    "input/graphTriangleCount/twitter_graph_v2.txt temp/GraphTriangleCount");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "GraphTriangleCount");
        job.setJarByClass(GraphTriangleCount.class);
        job.setMapperClass(GraphTriangleCountMapper.class);
        job.setReducerClass(GraphTriangleCountReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);
        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        SequenceFileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        SequenceFileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        job.setNumReduceTasks(TriangleCountDriver.ReducerNum);
        if (!job.waitForCompletion(true)) {
            System.exit(1);
        }
    }

    public static class GraphTriangleCountMapper extends Mapper<IntWritable, IntWritable, IntWritable, IntWritable> {
        @Override
        public void map(IntWritable key, IntWritable value, Context context)
                throws IOException, InterruptedException{
            context.write(key, value);
        }
    }

    public static class GraphTriangleCountReducer extends Reducer<IntWritable, IntWritable, Text, LongWritable> {
        private final static String edgePath = TriangleCountDriver.HDFS_PATH + TriangleCountDriver.EdgeConvertPath;

        private Map<Integer, Integer> vexIndex;
        private ArrayList<ArrayList<Integer>> vec = new ArrayList<ArrayList<Integer>>();
        private long triangleNum = 0;

        @Override
        public void setup(Context context)
                throws IOException, InterruptedException {
            int cnt = 0;
            int lastVertex = -1;
            ArrayList<Integer> outVertices = new ArrayList<Integer>();
            vexIndex = new TreeMap<Integer, Integer>();
            //获取文件系统的接口
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);
            for (FileStatus fst: fs.listStatus(new Path(edgePath))) {
                if (!fst.getPath().getName().startsWith("_")) {
                    SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(fst.getPath()));
                    IntWritable key = new IntWritable();
                    IntWritable value = new IntWritable();
                    while (reader.next(key, value)) {
                        if (key.get() != lastVertex) {
                            if (cnt != 0) vec.add(outVertices);
                            vexIndex.put(key.get(), cnt);
                            cnt++;
                            outVertices = new ArrayList<Integer>();
                            outVertices.add(value.get());
                        } else {
                            outVertices.add(value.get());
                        }
                        lastVertex = key.get();
                    }
                    reader.close();
                }
            }
            vec.add(outVertices);
        }

        @Override
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException{
            for (IntWritable val: values)
                if (vexIndex.containsKey(val.get()))
                    triangleNum += intersect(vec.get(vexIndex.get(key.get())), vec.get(vexIndex.get(val.get())));
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException{
            context.write(new Text("TriangleNum"), new LongWritable(triangleNum));
        }

        private long intersect(ArrayList<Integer> avex, ArrayList<Integer> bvex) {
            long num = 0;
            int i = 0, j = 0;
            int cv;
            while (i != avex.size() && j != bvex.size()) {
                cv = avex.get(i) - bvex.get(j);
                if (cv > 0) {
                    j++;
                } else if (cv < 0) {
                    i++;
                } else {
                    i++;
                    j++;
                    num++;
                }
            }
            return num;
        }
    }
}
