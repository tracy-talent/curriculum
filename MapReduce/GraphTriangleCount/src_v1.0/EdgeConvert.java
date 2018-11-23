package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
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

    public static class EdgeConvertMapper extends Mapper<Object, Text, Text, Text> {
        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            Text vertex1 = new Text(itr.nextToken());
            Text vertex2 = new Text(itr.nextToken());
            if (!vertex1.equals(vertex2)) {
                context.write(vertex1, vertex2);
                context.write(vertex2, vertex1);
            }
        }
    }

    public static class EdgeConvertReducer extends Reducer<Text, Text, Text, Text> {
        private final static String outDegreePath = TriangleCountDriver.HDFS_PATH +
                TriangleCountDriver.OutDegreeStatPath + "part-r-00000";
        private Map<String, Integer> degree;
        private Map<String, Boolean> edgeExisted;

        @Override
        public void setup(Context context)
                throws IOException, InterruptedException {
            degree = new HashMap<String, Integer>();

            //读取存储节点度的文件
            Configuration conf = context.getConfiguration();
            SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(new Path(outDegreePath)));
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
            Text vertex = new Text();
            List<String> outvertex = new ArrayList<String>();
            edgeExisted = new HashMap<String, Boolean>();  //记录已处理边以避免重复统计
            for (Text val: values) {
                if (!edgeExisted.containsKey(val.toString())) {
                    edgeExisted.put(val.toString(), true);
                    //比较边两个端点的度大小
                    if (degree.get(val.toString()) > degree.get(key.toString())) {
                        outvertex.add(val.toString());
                    }
                }
            }
            //对邻接节点从小到大进行排序，方便后续求交集
            Collections.sort(outvertex, new ComparatorString());
            for (String vt: outvertex) {
                vertex.set(vt);
                context.write(key, vertex);
            }
        }
    }

    //继承String比较器按它所表示的数值大小进行比较
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
