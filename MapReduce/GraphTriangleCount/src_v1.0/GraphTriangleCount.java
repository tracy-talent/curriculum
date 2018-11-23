package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.net.URI;
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
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
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

    public static class GraphTriangleCountMapper extends Mapper<Text, Text, Text, Text> {
        @Override
        public void map(Text key, Text value, Context context)
                throws IOException, InterruptedException{
            context.write(key, value);
        }
    }

    public static class GraphTriangleCountReducer extends Reducer<Text, Text, Text, LongWritable> {
        private final static String edgePath = TriangleCountDriver.HDFS_PATH + TriangleCountDriver.EdgeConvertPath;  //邻接表文件路径
        private Map<String, Integer> vexIndex;  //存储节点的邻接表索引
        private ArrayList<ArrayList<String>> vec = new ArrayList<ArrayList<String>>();  //存储全局邻接表
        private long triangleNum = 0;

        @Override
        public void setup(Context context)
                throws IOException, InterruptedException {
            int cnt = 0;
            String lastVertex = "";
            String sv, tv;
            ArrayList<String> outVertices = new ArrayList<String>();
            vexIndex = new TreeMap<String, Integer>();
            //获取文件系统的接口
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);
            //读取小度指向大度的边邻接表
            for (FileStatus fst: fs.listStatus(new Path(edgePath))) {
                if (!fst.getPath().getName().startsWith("_")) {
                    SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(fst.getPath()));
                    Text key = new Text();
                    Text value = new Text();
                    while (reader.next(key, value)) {
                        sv = key.toString();
                        tv = value.toString();
                        if (!sv.equals(lastVertex)) {
                            if (cnt != 0) vec.add(outVertices);
                            vexIndex.put(sv, cnt);
                            cnt++;
                            outVertices = new ArrayList<String>();
                            outVertices.add(tv);
                        } else {
                            outVertices.add(tv);
                        }
                        lastVertex = sv;
                    }
                    reader.close();
                }
            }
            vec.add(outVertices);
        }


        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException{
            for (Text val: values)
                if (vexIndex.containsKey(val.toString()))
                    //调用求交集函数获取包含边(key,val)的三角形个数
                    triangleNum += intersect(vec.get(vexIndex.get(key.toString())), vec.get(vexIndex.get(val.toString())));
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException{
            //将计数结果写入文件
            context.write(new Text("TriangleNum"), new LongWritable(triangleNum));
        }

        //求有序集合的交集
        private long intersect(ArrayList<String> avex, ArrayList<String> bvex) {
            long num = 0;
            int i = 0, j = 0;
            int cv;
            while (i != avex.size() && j != bvex.size()) {
                if (avex.get(i).length() > bvex.get(j).length()) {
                    cv = 1;
                } else if (avex.get(i).length() < bvex.get(j).length()) {
                    cv = -1;
                } else {
                    cv = avex.get(i).compareTo(bvex.get(j));
                }
                if (cv == 0) {
                    i++;
                    j++;
                    num++;
                } else if (cv > 0) {
                    j++;
                } else {
                    i++;
                }
            }
            return num;
        }
    }
}
