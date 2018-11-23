package main.java.TriangleCount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;

/**
 * 数据预处理包
 */
public class DataPreprocessUtil {

    //将离散化数据转换成序列化数据
    public static int Discrete2Sequence(Configuration conf, String inputPath, String outputPath) {
        Map<String, Integer> existedVertex = new HashMap<String, Integer>();
        int cnt = 0;

        try {
            FileSystem fs = FileSystem.get(new URI(inputPath), conf);
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(inputPath))));
            IntWritable key = new IntWritable();
            IntWritable value = new IntWritable();
            SequenceFile.Writer writer = SequenceFile.createWriter(conf, SequenceFile.Writer.file(new Path(outputPath)),
                    SequenceFile.Writer.keyClass(key.getClass()), SequenceFile.Writer.valueClass(value.getClass()));
            String line;
            String vertex1, vertex2;
            int a, b;
            while ((line = reader.readLine()) != null) {
                StringTokenizer itr = new StringTokenizer(line);
                vertex1 = itr.nextToken();
                vertex2 = itr.nextToken();
                if (!existedVertex.containsKey(vertex1)) {
                    existedVertex.put(vertex1, cnt);
                    a = cnt;
                    cnt++;
                } else {
                    a = existedVertex.get(vertex1);
                }
                if (!existedVertex.containsKey(vertex2)) {
                    existedVertex.put(vertex2, cnt);
                    b = cnt;
                    cnt++;
                } else {
                    b = existedVertex.get(vertex2);
                }
                key.set(a);
                value.set(b);
                writer.append(key, value);
            }
            reader.close();
            writer.close();
        }catch (IOException e) {
            e.printStackTrace();
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        return cnt;
    }
}
