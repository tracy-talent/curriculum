<center><h1>基于mapreduce实现文档倒排索引</h1></center>

## 一、程序的设计与实现

### 1.1 Map的设计与实现

* Map的设计目的是希望将词映射到出现频次，所以Map的key类型为Text，用于存储词和文件名，value类型为IntWritable，用于存储出现频次。在覆盖Mapper类中的map函数，将文本中的每个Text类型的==词#文件名==都映射到==IntWritable类型的1==，map函数的具体实现代码如下：

```java
 @Override
    public void map(Object key, Text value, Context context)
        throws IOException, InterruptedException {
        //获取当前输入的文件名
        FileSplit fileSplit = (FileSplit)context.getInputSplit();
        String fileName = fileSplit.getPath().getName();
        int spos = fileName.indexOf(".txt.segmented");
        if (spos == -1) {
            spos = fileName.indexOf(".TXT.segmented");
        }
        fileName = fileName.substring(0, spos);
        String temp;
        String line = value.toString().toLowerCase();
        StringTokenizer itr = new StringTokenizer(line);
        while (itr.hasMoreTokens()) {
            temp = itr.nextToken();
            if (!stopwords.contains(temp)) {
                //写入”词#文件名“到1的映射
                Text word = new Text();
                word.set(temp + "#" + fileName);
                context.write(word, one);
            }
        }
    }
```

* 为了去除一些出现频率高但是对文章信息挖掘又没什么帮助的词语，可以加入停词表来屏蔽这些词语，可以建一个集合，然后在重写Mapper类中的setup方法来加载停词表中的词语，setup函数的具体实现代码如下：

```java
@Override
    public void setup(Context context) throws IOException, InterruptedException {
        stopwords = new TreeSet<String>();
        //获取文件系统的接口
        Configuration conf = context.getConfiguration();
        FileSystem fs = FileSystem.get(conf);
        //获取缓存的文件
        cacheFiles = context.getCacheFiles();
        for (int i = 0; i < cacheFiles.length; i++) {
            String line;
            Path path = new Path(cacheFiles[i]);
            //调用FileSystem的接口来打开缓存的停词文件
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));
            while ((line = reader.readLine()) != null) {
                StringTokenizer itr = new StringTokenizer(line);
                while (itr.hasMoreTokens()) {
                    stopwords.add(itr.nextToken());
                }
            }
            reader.close();
        }
    }
```

### 1.2 Combiner的设计与实现

* Combiner相当于预reduce，在reduce之前对map输出数据的value进行累加，从而压缩map的输出数据，减少数据传输到Reducer的通信负载，具体的实现代码如下：

```java
@Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
        int sum = 0;
        for (IntWritable val: values) {
            sum += val.get();
        }
        result.set(sum);
        context.write(key, result);
    }
```

### 1.3 Partitioner的设计与实现

* 为了将不同文档中的同一词分配到相同reducer下面，需要对Mapper输出结果的==key:词#文件名==进行分区，让系统按key中的词进行数据分配，具体实现代码如下：

```java
public class DocInvertedIndexPartitioner extends HashPartitioner<Text, IntWritable> {
    @Override
    public int getPartition(Text key, IntWritable value, int numReduceTsaks) {
        String term = key.toString().split("#")[0];
        return super.getPartition(new Text(term), value, numReduceTsaks);
    }
}
```

### 1.4 Reducer的设计与实现

* Reducer的设计用于接收来自Mapper的==(Text,IntWritable)==数据，然后累加同一词语在不同文档中出现的频次，再通过公式(1)计算词语的平均出现次数，用String.format将平均出现次数格式化保留两位小数，然后将键值对==(词:Text，平均出现次数:Text)==写入到结果中。
$$
平均出现次数=\frac{词语在全部文档中出现的频数总和}{包含该词语的文档数}　　　(1)
$$

* Reducer的reduce和cleanup函数具体实现代码如下：

```java
 	/**
     * 计算每个词的评剧出现次数并写入结果
     * @param key
     * @param values
     * @param context
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
        int sum = 0;
        word1.set(key.toString().split("#")[0]);  //分割获取词
        String bookname = key.toString().split("#")[1];   //分割获取书名
        for (IntWritable val: values) {
            //对词在当前文档中的出现次数进行累加
            sum += val.get();
        }
        word2.set(bookname + ":" + sum);

        if (!currentWord.equals(word1) && currentWord.getLength() != 0) {
            StringBuilder out = new StringBuilder();
            long count = 0;
            //对词在不同文档中的出现次数进行累加
            for (String item: postingList) {
                out.append(item);
                out.append(";");
                count += Long.parseLong(item.substring(item.indexOf(":") + 1));
            }
            //格式化平均出现次数保留两位小数，并将词在每个文档中的出现频次增添到value中
            StringBuilder out1 = new StringBuilder(String.format("%.2f", (double)count/postingList.size()) + ",")
                    .append(out);
            if (count > 0) {
                //去除行尾的分号，然后写入reduce的键值对结果<Text,Text>
                context.write(currentWord, new Text(out1.substring(0, out1.lastIndexOf(";"))));
            }
            postingList = new ArrayList<String>();
        }
        //attention:currentWord=word1是把引用赋值给currentWord，word1变化会导致currentWord变化，所以要用new Text
        currentWord = new Text(word1);
        postingList.add(word2.toString());
    }

    /**
     * 计算当前reducer中最后一个词的平均出现次数并写入结果
     * @param context
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void cleanup(Context context)
        throws IOException, InterruptedException {
        StringBuilder out = new StringBuilder();
        long count = 0;
        for (String item: postingList) {
            out.append(item);
            out.append(";");
            count += Long.parseLong(item.substring(item.indexOf(":") + 1));
        }
        StringBuilder out1 = new StringBuilder(String.format("%.2f", (double)count/postingList.size()) + ",")
                .append(out);
        if (count > 0) {
            context.write(currentWord, new Text(out1.substring(0, out1.lastIndexOf(";"))));
        }
    }
```



## 二、扩展

### 2.1 扩展一

扩展任务描述：使用另外一个 MapReduce Job 对每个词语的平均出现次数进行全局排序,
输出排序后的结果。

- 可以在上面实验结果的基础上稍作修改即可，将上面实验的reduce输出的==(词：Text，平均出现次数,文档:频次...:Text)==键值对改为输出==(词:Text,平均出现次数:DoubleWritable)==，将结果写入到临时文件，然后新建一个MapReudce Job，使用==InvertedMapper==类将前一个Job的输出结果做逆映射，然后继承==DouWritable.Comparator==构造降序排序的比较器并将其作为当前这个Job的排序比较器。

### 2.2 扩展二

扩展任务描述：为每位作家、计算每个词语的 TF-IDF。TF 定义为某个词语在某个作家的
所有作品中的出现次数之和。IDF 定义为:
$$
IDF(词语)=log(\frac{语料库文档总数}{包含该词的文档数 + 1})
$$
输出格式:作家名字,词语,该词语的 TF-IDF。

- 这个扩展在第一个实验的基础上稍作修改即可达到要求，将Mapper的key由==词#文件名==改为==作者@词#文件名==，类型不变依然为Text，然后将Partitioner由按key中的==词==分区改为按==作者@词==分区。而对于IDF计算公式中的语料库中总文档数的获取可以通过hdfs提供文件系统访问接口得到，重写Reducer类中的setup方法，代码如下：

```java
    @Override
    public void setup(Context context)
            throws IOException, InterruptedException{
        FileSystem fs = FileSystem.get(context.getConfiguration());
        fileNum = fs.listStatus(new Path("hdfs://192.168.1.1:9000/data/wuxia_novels")).length;
    }
```