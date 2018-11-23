package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.util.*;

public class UndirectionalEdgeConvert {

    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        conf.set("mapreduce.reduce.memory.mb", "2048");  //设置reduce container的内存大小
        conf.set("mapreduce.reduce.java.opts", "-Xmx2048m");  //设置reduce任务的JVM参数
//        conf.set("mapreduce.map.java.opts", "-Xmx2048m");  //设置map任务的JVM参数

        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: hadoop jar UndirectionalEdgeConvert.jar UndirectionalEdgeConvert " +
                    "input/graphTriangleCount/twitter_graph_v2.txt temp/UndirectionalEdgeConvert");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "UndirectionalEdgeConvert");
        job.setJarByClass(UndirectionalEdgeConvert.class);
        job.setMapperClass(UndirectionalEdgeConvertMapper.class);
        job.setReducerClass(UndirectionalEdgeConvertReducer.class);
        job.setPartitionerClass(UndirectionalEdgeConvertPartitioner.class);
        job.setMapOutputKeyClass(MyLongWritable.class);
        job.setMapOutputValueClass(ByteWritable.class);
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

    public static class UndirectionalEdgeConvertMapper extends Mapper<IntWritable, IntWritable, MyLongWritable, ByteWritable> {
        @Override
        public void map(IntWritable key, IntWritable value, Context context)
                throws IOException, InterruptedException {
            /**
             * 优化点：用long来代替Text存储节点的键值对，前提是节点数目大小在INT_MAX以内(对于
             * 离散化数值很大的节点可以把它们顺序化编号再用此方法处理)，key存储在高32位，value存储在低32位
             */
            long key1 = key.get();
            key1 = (key1 << 32) + value.get();
            long key2 = value.get();
            key2 = (key2 << 32) + key.get();
            context.write(new MyLongWritable(key1), new ByteWritable());
            context.write(new MyLongWritable(key2), new ByteWritable());
        }
    }

    public static class UndirectionalEdgeConvertReducer extends Reducer<MyLongWritable, ByteWritable, IntWritable, IntWritable> {
        private final static String outDegreePath = TriangleCountDriver.HDFS_PATH +
                TriangleCountDriver.OutDegreeStatPath + "part-r-00000";

        private Map<Integer, Integer> degree;
        private List<Integer> outvertex;
        private int lastKey;

        @Override
        public void setup(Context context)
                throws IOException, InterruptedException {
            lastKey = -1;
            outvertex = new ArrayList<Integer>();
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
        public void reduce(MyLongWritable key, Iterable<ByteWritable> values, Context context)
                throws IOException, InterruptedException{
            int a = (int)(key.get() >>> 32);  //逻辑右移取high 32 bit
            int b = (int)(key.get());  //直接截取low 32 bit
            if (lastKey != a) {
                if (outvertex.size() != 0) {
                    IntWritable vertex = new IntWritable();
                    IntWritable lastKeyWr = new IntWritable(lastKey);
                    Collections.sort(outvertex);
                    for (Integer vt: outvertex) {
                        vertex.set(vt);
                        context.write(lastKeyWr, vertex);
                    }
                    outvertex = new ArrayList<Integer>();
                }
            }

            int cnt = 0;
            for (ByteWritable val: values)
                cnt++;
            if (cnt == 2) {
                if (degree.get(b) > degree.get(a)) {
                    outvertex.add(b);
                }
            }
            lastKey = a;
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            if (outvertex.size() != 0) {
                IntWritable vertex = new IntWritable();
                IntWritable lastKeyWr = new IntWritable(lastKey);
                Collections.sort(outvertex);
                for (Integer vt: outvertex) {
                    vertex.set(vt);
                    context.write(lastKeyWr, vertex);
                }
            }
        }
    }

    public static class UndirectionalEdgeConvertPartitioner extends HashPartitioner<MyLongWritable, ByteWritable> {
        @Override
        public int getPartition(MyLongWritable key, ByteWritable value, int numReduceTsaks) {
            return super.getPartition(new MyLongWritable(key.get() >>> 32), value, numReduceTsaks);
        }
    }

    public static class MyLongWritable extends LongWritable {

        public MyLongWritable(){}

        public MyLongWritable(long value){
            super(value);
        }

        @Override
        public int compareTo(LongWritable o) {
            long a = this.get() >> 32;
            long b = o.get() >> 32;
            if (a > b) {
                return 1;
            } else if (a < b) {
                return -1;
            } else {
                return super.compareTo(o);
            }
        }
    }
}
