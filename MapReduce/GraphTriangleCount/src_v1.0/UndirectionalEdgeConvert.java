package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.net.URI;
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
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        SequenceFileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        job.setNumReduceTasks(TriangleCountDriver.ReducerNum);
        if(!job.waitForCompletion(true)) {
            System.exit(1);
        }
    }

    public static class UndirectionalEdgeConvertMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            String vertex1 = itr.nextToken();
            String vertex2 = itr.nextToken();
            String flag = "0";  //节点序标记，帮助Reducer判断双向边
            if (!vertex1.equals(vertex2)) {
                if (vertex1.compareTo(vertex2) > 0) flag = "1";
                context.write(new Text(vertex1 + '\t' + vertex2), new Text(flag));
                context.write(new Text(vertex2 + '\t' + vertex1), new Text(flag));
            }
        }
    }

    public static class UndirectionalEdgeConvertReducer extends Reducer<Text, Text, Text, Text> {
        private final static String outDegreePath = TriangleCountDriver.HDFS_PATH +
                TriangleCountDriver.OutDegreeStatPath + "part-r-00000";
        private Map<String, Integer> degree;
        private List<String> outvertex;
        private String lastKey;
        private URI[] cacheFiles;

        @Override
        public void setup(Context context)
                throws IOException, InterruptedException {
            lastKey = "";
            outvertex = new ArrayList<String>();
            degree = new HashMap<String, Integer>();

            //读取节点的度文件
            Configuration conf = context.getConfiguration();
            SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(new Path(outDegreePath)));
//            Writable key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), fs.getConf());
//            Writable value = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), fs.getConf());
            IntWritable key = new IntWritable();
            Text value = new Text();
            int cnt = 0;
            while (reader.next(key, value)) {
                degree.put(value.toString(), cnt);
                cnt++;
            }
            reader.close();
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException{
            String[] term = key.toString().split("\t");
            //如果邻接表出点变了则写出lastkey为出点的邻接表
            if (!lastKey.equals(term[0])) {
                if (outvertex.size() != 0) {
                    Text vertex = new Text();
                    Text lastKeyText = new Text(lastKey);
                    Collections.sort(outvertex, new EdgeConvert.ComparatorString());
                    for (String vt: outvertex) {
                        vertex.set(vt);
                        context.write(lastKeyText, vertex);
                    }
                    outvertex = new ArrayList<String>();
                }
            }

            //判断(term[0], term[1])是否是双向边
            boolean flag0 = false, flag1 = false;
            for (Text val: values) {
                if (val.toString().equals("0")) flag0 = true;
                else flag1 = true;
            }
            if (flag0 && flag1) {
                if (degree.get(term[1]) > degree.get(term[0])) {
                    outvertex.add(term[1]);
                }
            }
            lastKey = term[0];
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            //写出最后一个出点的邻接表
            if (outvertex.size() != 0) {
                Text vertex = new Text();
                Text lastKeyText = new Text(lastKey);
                Collections.sort(outvertex, new EdgeConvert.ComparatorString());
                for (String vt: outvertex) {
                    vertex.set(vt);
                    context.write(lastKeyText, vertex);
                }
            }
        }
    }

    public static class UndirectionalEdgeConvertPartitioner extends HashPartitioner<Text, Text> {
        @Override
        public int getPartition(Text key, Text value, int numReduceTsaks) {
            String term = key.toString().split("\t")[0];
            return super.getPartition(new Text(term), value, numReduceTsaks);
        }
    }
}
