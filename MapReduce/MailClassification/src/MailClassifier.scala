import com.esotericsoftware.kryo.Kryo
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, Word2Vec}
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import java.util.concurrent.TimeUnit

class MyRegistrator extends KryoRegistrator {
    override def registerClasses(kryo: Kryo) {
        kryo.register(classOf[scala.collection.mutable.HashSet[String]])
    }
}

object MailClassifier {
    // Word2Vec超参
    final val W2V_MAX_ITER = 5
    final val EMBEDDING_SIZE = 128
    final val MIN_COUNT = 1 // default: 5
    final val WINDOW_SIZE = 5  // default: 5
    // MLP超参
    final val MLP_MAX_ITER = 300
    final val BLOCK_SIZE = 128  // default:128
    final val SEED = 1234L
    final val HIDDEN1_SIZE = 64
    final val HIDDEN2_SIZE = 32
    final val LABEL_SIZE = 20

    def main(args: Array[String]): Unit = {
        var timeUsed = System.nanoTime()
        val startTime = System.nanoTime()
        //close log
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val conf = new SparkConf().setAppName("frequent itemset mining")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.kryo.registrator", "MyRegistrator")
                .set("spark.broadcast.factory", "org.apache.spark.broadcast.TorrentBroadcastFactory")
                .set("spark.storage.memoryFraction", "0.4")
                .set("spark.shuffle.memoryFraction", "0.3")
                .set("spark.shuffle.consolidateFiles", "true")
                .setAppName("MailClassifier")

        val spark = SparkSession.builder().config(conf).getOrCreate()
        val sc = spark.sparkContext
        if(args.length < 2){
            System.err.println("Usage: <inputDataset> <outputDir>")
            System.exit(-1)
        }
        // 接收命令行传递的参数(6个参数)
        val input = args(0) //输入文件
        val output = if(args(1).endsWith("/")) args(1).substring(0, args(1).length() - 1) else args(1) //输出目录(会覆盖)

        // 检测输出目录是否存在，存在则删除
        val hdfsConf = new Configuration()
        val hdfs = FileSystem.get(hdfsConf)
        val path : Path = new Path(output)
        if (hdfs.exists(path)) {
            hdfs.delete(path, true)
        }

        val parsedRDD = sc.textFile(input).map(_.split("\t")).map(eachrow => {
            (eachrow(0), eachrow(1).split(" "))
        })
        val mailDF = spark.createDataFrame(parsedRDD).toDF("label", "message")
        val labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(mailDF)

        val word2Vec = new Word2Vec()
                .setMaxIter(W2V_MAX_ITER)
                .setVectorSize(EMBEDDING_SIZE)
                .setMinCount(MIN_COUNT)
                .setWindowSize(WINDOW_SIZE)
                .setInputCol("message")
                .setOutputCol("features")

        val layers = Array[Int](EMBEDDING_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, LABEL_SIZE)
        val mlpc = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setMaxIter(MLP_MAX_ITER)
                .setBlockSize(BLOCK_SIZE)
                .setSeed(SEED)
                .setFeaturesCol("features")
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")

        val labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels)

        val Array(trainingData, testData) = mailDF.randomSplit(Array(0.8, 0.2))

        val pipeline = new Pipeline().setStages(Array(labelIndexer, word2Vec, mlpc, labelConverter))
        val model = pipeline.fit(trainingData)

        val predictionResultDF = model.transform(testData)
        predictionResultDF.select( "label", "predictedLabel")
                .write.format("csv").save(output)
        // below 2 lines for debug
        predictionResultDF.printSchema()
        predictionResultDF.select("message", "label", "predictedLabel").show(30)

        val evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy")
        val predictionAccuracy = evaluator.evaluate(predictionResultDF)
        println("Testing Acurracy is %2.4f".format(predictionAccuracy * 100) + "%")
        evaluator.setMetricName("f1")
        val predictionF1 = evaluator.evaluate(predictionResultDF)
        println("Testing F1 is %2.4f".format(predictionF1 * 100) + "%")
        timeUsed = System.nanoTime() - timeUsed
        println("preprocess used " + TimeUnit.SECONDS.convert(timeUsed, TimeUnit.NANOSECONDS) + " s.")
//        StdIn.readLine()
        spark.stop()
    }
}
