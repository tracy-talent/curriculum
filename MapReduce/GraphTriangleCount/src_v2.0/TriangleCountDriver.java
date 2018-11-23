package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.GenericOptionsParser;

import java.net.URI;

/**
 * a->b or b->a then a-b
 */
public class TriangleCountDriver {
    public final static String HDFS_PATH = "hdfs://192.168.100.103:9000/user/liujian/";
    public final static String OutDegreeStatPath = "temp/OutDegreeStat/";
    public final static String EdgeConvertPath = "temp/EdgeConvert/";
    public final static String GraphTriangleCountPath = "temp/GraphTriangleCount/";
    public final static int ReducerNum = 6;

    private final static String childResPath = HDFS_PATH + GraphTriangleCountPath;

    public static int VerTexSum;

    public static void main(String[] args) throws Exception{
        long elapseTime = System.currentTimeMillis();

        Configuration conf = new Configuration();
        conf.set("mapreduce.reduce.memory.mb", "2048");  //设置reduce container的内存大小
        conf.set("mapreduce.reduce.java.opts", "-Xmx2048m");  //设置reduce任务的JVM参数
//        conf.set("mapreduce.map.java.opts", "-Xmx2048m");  //设置map任务的JVM参数

        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 2) {
            System.err.println("Usage: hadoop jar TriangleCount.jar TriangleCountDriver " +
                    "input/graphTriangleCount/twitter_graph_v2.txt output");
            System.exit(2);
        }

        /**
         * 将输入文件中离散化节点转换成顺序的，并存储成sequencefile
         * 目的：
         * 1. 序列化之后节点大小控制在节点数目以内，因此可用int代替text存储，避免在reduce阶段对字符串排序
         * 2. hadoop需要将数据序列化之后才能在网络上传输，存成sequencefile以减少数据格式转换的时间开销
         * 缺点：
         * 对数据进行转换的预处理耗时，第一生成会很慢，整体时间开销比不转换数据的做法还慢，
         * 存起来后续再次运行直接使用，整体性能就有提升了
         * 测试：
         * 1.gplus: 240s
         * 2.twitter: 135s
         */
//        String inputPath = HDFS_PATH + otherArgs[0];
//        String inputTag[] = otherArgs[0].split("/");
//        String inputTransPath = HDFS_PATH + inputTag[inputTag.length - 1].charAt(0) + "input_trans_file";
//        VerTexSum = DataPreprocessUtil.Discrete2Sequence(conf, inputPath, inputTransPath);
//        System.out.println("顶点数： " + VerTexSum);
//        otherArgs[0] = inputTransPath;

        String[] forGB = {otherArgs[0], ""};

        forGB[1] = OutDegreeStatPath;
        OutDegreeStat.main(forGB);

        forGB[1] = EdgeConvertPath;
        EdgeConvert.main(forGB);

        forGB[0] = EdgeConvertPath;
        forGB[1] = GraphTriangleCountPath;
        GraphTriangleCount.main(forGB);

        String outputPath = HDFS_PATH + otherArgs[1] + "/part-r-00000";
        long triangleSum = 0;
        FileSystem fs = FileSystem.get(new URI(childResPath), conf);
        for (FileStatus fst: fs.listStatus(new Path(childResPath))) {
            if (!fst.getPath().getName().startsWith("_")) {
                SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(fst.getPath()));
                Text key = new Text();
                LongWritable value = new LongWritable();
                reader.next(key, value);
                triangleSum += value.get();
                reader.close();
            }
        }
        fs = FileSystem.get(new URI(outputPath), conf);
        FSDataOutputStream outputStream = fs.create(new Path(outputPath));
        outputStream.writeChars("TriangleSum = " + triangleSum + "\n");
        outputStream.close();
        elapseTime = System.currentTimeMillis() - elapseTime;
        System.out.println("TriangleSum = " + triangleSum);
        System.out.println("Timeused: " + elapseTime/1000 + "s");
//        //删除生成的输入文件SequenceFile版本
//        fs.delete(new Path(inputTransPath), true);
        System.exit(0);
    }
}
