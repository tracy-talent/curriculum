<center><h1>MapReduce实验四：图的三角形计数</h1></center>

<center>组号：2018st32&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;日期：2018/11/22</center>



## 一、实验设计与实现

### 1.1 算法设计

* step1：统计图中每一个点的度，不关心是入度还是出度，然后对统计到的所有点的度进行排序
* step2：将图中每一条单向边转换成双向边，对于图中a->b and b->a的两条边，分别转换后需要去重，在转换后的图中筛选出小度指向大度的边来建立邻接表，然后对每个点的邻接点按从小到大进行排序
* step3：对原图中的边进行转换，确保每条边是由数值小的点指向数值大的点并去重，然后遍历每一条边：求边的两个端点对应的邻接点集的交集大小即为包含这条边的三角形个数。对每条边对应的三角形个数进行累加即可得到全图包含的三角形个数。



### 1.2 程序设计

* 根据算法步骤将程序设计成4个job来实现：
  1. job:OutDegreeStat用于对每个点的度进行统计，在类OutDegreeStat中实现
  2. job:SortedOutDegree用于对所有点的度进行排序，在类OutDegreeStat中实现，在job1之后运行
  3. job:EdgeConvert用于建立存储小度指向大度的边的邻接表，在类EdgeConvert中实现
  4. job:GraphTriangleCount用于遍历每条边求端点对应邻接点集的交集来对三角形进行计数，在类GraphTriangleCount中实现



### 1.3 程序实现

* job:OutDegreeStat的实现：

  1. Map阶段：(vertex1: Text, vertex2: Text) -> (vertex1: Text, 1: IntWritable) and (vertex2: Text, 1: IntWritable)，实现代码如下：

     ```java
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
     ```

  2. Reduce阶段：(vertex: Text, degree: Iterable\<IntWritable\>) -> (vertex: Text, degreeSum: IntWritable)，实现代码如下：

     ```java
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
     ```

  3. Combiner阶段：Combiner逻辑与Reduce逻辑一致，只是为了减少数据量从而减少通信开销



* job:SortedOutDegree的实现：
  1. Map阶段：由于mapreduce的reduce阶段会按key进行排序，为了按度进行排序，只需用hadoop自带的InverseMapper类对键值对做逆映射(vertex: Text, degree: IntWritable) -> (degree: IntWritable, vertex: Text)即可。
  2. Reduce阶段：无需设置Reducer类，hadoop的Reduce阶段自动会对degree进行排序



* job:EdgeConvert的实现：

  1. Map阶段：(vertex1: Text, vertex2: Text) -> (vertex1: Text, vertex2: Text) and (vertex2: Text, vertex1: Text)，实现代码如下：

     ```java
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
     ```

  2. Reduce阶段：在setup函数中读取存储节点度的文件，在reduce函数中(vertex1: Text, vertex2List: iterable\<Text\>) -> (vertex1 with minimal degree: Text, vertex2 with maximal degree: Text)，==在对邻接表节点进行排序时，要重写一个String Comparator，让String按它所表示的数值大小进行比较，==实现代码如下：

     ```java
     public static class EdgeConvertReducer extends Reducer<Text, Text, Text, Text> {
             private Map<String, Integer> degree;
             private Map<String, Boolean> edgeExisted;
             private URI[] cacheFiles;
     
             @Override
             public void setup(Context context)
                     throws IOException, InterruptedException {
                 degree = new HashMap<String, Integer>();
     
                 //读取存储节点度的文件
                 Configuration conf = context.getConfiguration();
                 cacheFiles = context.getCacheFiles();
                 for (int i = 0; i < cacheFiles.length; i++) {
                     SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(new Path(cacheFiles[i])));
                     IntWritable key = new IntWritable();
                     Text value = new Text();
                     int cnt = 0;
                     while (reader.next(key, value)) {
                         degree.put(value.toString(), cnt);
                         cnt++;
                     }
                     reader.close();
                 }
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
     ```



* job:GraphTriangleCount的实现：

  1. Map阶段：以job:EdgeConvert的输出文件作为读入，该文件不包含重复边因此无需判断转换，直接按原样映射(vertex1: Text, vertex2: Text) -> (vertex1: Text, vertex2: Text)即可，实现代码如下：

     ```java
     public static class GraphTriangleCountMapper extends Mapper<Text, Text, Text, Text> {
             @Override
             public void map(Text key, Text value, Context context)
                     throws IOException, InterruptedException{
                 context.write(key, value);
             }
         }
     ```

  2. Reduce阶段：在setup函数中读取存储小度指向大度的邻接表文件，在reduce函数中(vertex1: Text, vertex2List: Iterable\<Text\>) -> ("TriangleNum": Text, triangleNum: LongWritable)，在cleanup函数中写当前这个reducer的三角形计数结果，实现代码如下：

     ```java
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
     ```



### 1.4 扩展任务1的设计与实现

* 对于a->b and b->a then a-b的条件，只需改变job:EdgeConvert的实现即可，新建一个job:UndirectionalEdgeConvert:

  1. Map阶段：以原数据文件作为输入进行映射转换：(vertex1: Text, vertex2: Text) -> ((vertex1, vertex2): Text, flag: Text) and ((vertex2, vertex1): Text, flag:Text)，按vertex1进行分区，其中flag作为节点序标记，用于帮助在Reducer中对双向边进行判断，如果输入的vertex1 > vertex2则flag置1，如果vertex1 < vertex2则flag置0，实现代码如下：

     ```java
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
     ```

  2. Reduce阶段：((vertex1, vertex2): Text, flagList: Iterable\<Text\>) -> (vertex1 with minimal degree: Text, vertex2 with maximal degree: Text)，如果flagList既包含0又包含1，则(vertex1, verex2)属于双向边，然后判断如果vertex2的度大于vertex1，则将vertex2加入到vertex1的邻接表中，reduce函数部分的实现代码如下：

     ```java
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
     ```



## 二、程序运行方式及结果

### 2.1 jar包运行方式

==每次运行jar包之前先执行hadoop fs -rmr temp/*删除生成的中间文件==

* condition: if a->b or b->a then a-b

  * data: twitter_graph_v2.txt

    ```shell
    hadoop jar TriangleCount.jar main.java.TriangleCount.TriangleCountDriver /data/graphTriangleCount/twitter_graph_v2.txt toutput
    ```

  * data: gplus_combined.unique.txt

    ```shell
    hadoop jar TriangleCount.jar main.java.TriangleCount.TriangleCountDriver /data/graphTriangleCount/gplus_combined.unique.txt goutput
    ```

* condition: if a->b and b->a then a-b

  * data: twitter_graph_v2.txt

    ```shell
    hadoop jar TriangleCount.jar main.java.TriangleCount.UndirectionalTriangleCountDriver /data/graphTriangleCount/twitter_graph_v2.txt undir-toutput
    ```

  * data: gplus_combined.unique.txt

    ```shell
    hadoop jar TriangleCount.jar main.java.TriangleCount.UndirectionalTriangleCountDriver /data/graphTriangleCount/gplus_combined.unique.txt undir-goutput
    ```



### 2.2 运行结果及时耗

=="or"表示if a->b or b->a then a-b的情况，"and"表示if a->b and b->a then a-b的情况==

| 数据集       | 三角形个数 | Driver程序在集群上的运行时间(秒) |
| ------------ | ---------- | -------------------------------- |
| Twitter(or)  | 13082506   | 127s                             |
| Google+(or)  | 1073677742 | 308s                             |
| Twitter(and) | 1818304    | 157s                             |
| Goolge+(and) | 27018510   | 216s                             |



### 2.3  输出结果信息

* condition: if a->b or b->a then a-b

  * data: twitter_graph_v2.txt

    ```s
    输出结果路径：/usr/2018st32/t10-output/part-r-00000
    ```

  * gplus_combined.unique.txt

    ```
    输出结果路径：/usr/2018st32/g10-output/part-r-00000
    ```

* condition: if a->b and b->a then a-b

  * data: twitter_graph_v2.txt

    ```
    输出结果路径：/usr/2018st32/t10-undir-output/part-r-00000
    ```

  * gplus_combined.unique.txt

    ```
    输出结果路径：/usr/2018st32/g10-undir-output/part-r-00000
    ```

* 下面是上面四个结果路径的截图：

![](/home/brooksj/图片/2018-11-23 14-23-34 的屏幕截图.png)



## 三、性能分析与优化

### 3.1 性能分析

* 该算法的性能瓶颈在遍历每一条边然后边两个端节点对应邻接表的交集，然后对每个顶点出发的邻接表进行排序也比较耗时，算法整体的时间复杂度是O(E^1.5^)，E为边的数目
* 目前这个1.0版本的实现鲁棒性比较好，节点编号用Text存储，所以无论节点编号多大都可以存储以及比较，输入的图可以允许重复边的出现，不会影响结果的正确性。但由于mapreduce涉及大量的排序过程，用Text存储节点也就意味着使用字符串排序，字符串之间的比较当然比整型比较开销大，从而会影响程序的整体性能。除此之外，hadoop需要对数据进行序列化之后才能在网络上传输，数据以文本文件输入导致大量的数据序列化转换也会降低程序性能。



### 3.2 性能优化

2.0版本，在1.0的版本上进行了数据储存和表示方面的优化，相同实验环境(6个Reducer，每个Reducer2G物理内存，Reducer中的java heapsize -Xmx2048m)跑Goolge+数据能快50s左右，具体优化细节如下：

* 将离散化稀疏的节点转换成顺序化的，这样就可以用IntWritabel表示节点(前提是节点数未超过INT_MAX)而不用Text来表示节点编号，这样就可以避免字符串排序，减少map和reduce阶段的排序开销
* 将原始Text输入文件转换成Sequence，因为hadoop传输在网络上的数据是序列化的，这样可以避免数据的序列化转换开销。但是由于数据是串行转换的，影响整体性能，但是可以在第一次运行过后存起来，以后运行直接加载sequence的数据文件即可。这一步是和第一步顺序化节点一起完成的，转换后的sequence文件存储的是顺序化的节点表示的边。
* 在a->b and b->a then a-b的条件下，在获取小度指向大度的边集任务中，mapper需要将一条边的点对合并为key以在reducer中判断是否是双向边，看似只能用Text来存储了，实则这里有一个trick，在对节点顺序化之后的节点数通常不会超过INT_MAX，因此可以使用考虑将两个int型表示的节点转换成long,key存储在高32位，value存储在低32位，通过简单的位操作即可实现，这样mapper输出的key就是long而非Text，从而避免了字符串的比较排序，由于mapreduce涉及大量排序过程，因此在涉及程序的时候尽量用一些trick避免用Text表示key.



## 四、WebUI执行报告

* Twitter(or)

  * job:OutDegreeStat

    ![](/home/brooksj/图片/2018-11-23 14-28-48 的屏幕截图.png)

  * job:SortedOutDegree

    ![](/home/brooksj/图片/2018-11-23 14-30-08 的屏幕截图.png)

  * job:EdgeConvert

    ![](/home/brooksj/图片/2018-11-23 14-31-19 的屏幕截图.png)

  * job:GraphTriangleCount

    ![](/home/brooksj/图片/2018-11-23 14-33-22 的屏幕截图.png)



* Google+(or)

  * job:OutDegreeStat

    ![](/home/brooksj/图片/2018-11-23 14-35-46 的屏幕截图.png)

  * job:SortedOutDegree

    ![](/home/brooksj/图片/2018-11-23 14-36-31 的屏幕截图.png)

  * job:EdgeConvert

    ![](/home/brooksj/图片/2018-11-23 14-37-29 的屏幕截图.png)

  * job:GraphTriangleCount

    ![](/home/brooksj/图片/2018-11-23 14-38-02 的屏幕截图.png)



* Twitter(and)

  * job:OutDegreeStat

    ![](/home/brooksj/图片/2018-11-23 14-44-32 的屏幕截图.png)

  * job:SortedOutDegree

    ![](/home/brooksj/图片/2018-11-23 14-51-27 的屏幕截图.png)

  * job:UndirectionalEdgeConvert

    ![](/home/brooksj/图片/2018-11-23 14-44-47 的屏幕截图.png)

  * job:GraphTriangleCount

    ![](/home/brooksj/图片/2018-11-23 14-45-23 的屏幕截图.png)



* Google+(and)

  * job:OutDegreeStat

    ![](/home/brooksj/图片/2018-11-23 14-45-35 的屏幕截图.png)

  * job:SortedOutDegree

    ![](/home/brooksj/图片/2018-11-23 14-45-53 的屏幕截图.png)

  * job:UndirectionalEdgeConvert

    ![](/home/brooksj/图片/2018-11-23 14-46-04 的屏幕截图.png)

  * job:GraphTriangleCount

    ![](/home/brooksj/图片/2018-11-23 14-46-21 的屏幕截图.png)