package main.scala.Apriori.distributed

import com.esotericsoftware.kryo.Kryo
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import java.util.concurrent.TimeUnit

import org.apache.hadoop.hdfs.server.blockmanagement.NumberReplicas
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.serializer.KryoRegistrator

import scala.tools.nsc.interpreter.Completion.Candidates
import scala.io.StdIn
import scala.collection.mutable.HashSet
import scala.collection.mutable.HashMap

class MyRegistrator extends KryoRegistrator {
    override def registerClasses(kryo: Kryo) {
        kryo.register(classOf[scala.collection.mutable.HashSet[String]])
    }
}


object Apriori {
    def main (args: Array[String]): Unit = {
        val startTime = System.nanoTime()
        //close log
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val conf = new SparkConf().setAppName("frequent itemset mining")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.kryo.registrator", "main.scala.Apriori.distributed.MyRegistrator")
                .set("spark.broadcast.factory", "org.apache.spark.broadcast.TorrentBroadcastFactory")
                .set("spark.storage.memoryFraction", "0.4")
                .set("spark.shuffle.memoryFraction", "0.3")
                .set("spark.shuffle.consolidateFiles", "true")

        val sc = new SparkContext(conf)

        if(args.length < 6){
            System.err.println("Usage: <inputDataset> <outputDir>")
            System.exit(-1)
        }
        // 接收命令行传递的参数(6个参数)
        val input = args(0) //输入文件
        val output = if(args(1).endsWith("/")) args(1).substring(0, args(1).length() - 1) else args(1) //输出目录(会覆盖)
        val support = args(2).toDouble  // 支持度
        val K = args(3).toInt  //迭代次数
        val num = args(4).toInt  // 并发度
        val confidence = args(5).toDouble  //置信度

        val infileName = input.substring(input.lastIndexOf('/')+1, input.lastIndexOf('.')) //输入文件名

        // 检测输出目录是否存在，存在则删除
        val hdfsConf = new Configuration()
        val hdfs = FileSystem.get(hdfsConf)
        val path : Path = new Path(output)
        if (hdfs.exists(path)) {
            hdfs.delete(path, true)
        }

        val ft0 = System.nanoTime()
        // 将输入数据分区，由于后面要频繁使用。因此缓存起来
        val transations = sc.textFile(input, num)
                .map(x => {
                    val content = x.split("\\s+")
                    val tmp = new HashSet[String]
                    for (i <- 0 until content.length) {
                        tmp.add(content(i))
                    }
                    tmp
                }).cache()

        // 计算频繁项阈值
        val numRecords = transations.count().toDouble
        val freqThre = numRecords * support
        println("frequency threshold：" + freqThre)
        // 计算频繁1项集
        val oneFreqSet = transations
                .flatMap(_.seq)
                .map(x => (x, 1))
                .reduceByKey(_ + _)
                .filter(x => x._2 >= freqThre)
                .map(x => (HashSet(x._1), x._2))
        // 输出频繁1项集到目录
        oneFreqSet.map(a => {
            val out = a._1.mkString(",") + ": " + a._2.toString + "  " + (a._2 / numRecords).toString
            out
        }).saveAsTextFile(output + "/" + infileName + "freqset-1")
        // 打印生成频繁1项集耗时
        val ft1 = System.nanoTime()
        val oneFT = TimeUnit.SECONDS.convert(ft1 - ft0, TimeUnit.NANOSECONDS)
        println("generate " + oneFreqSet.collect().length + " Frequent 1-Item Set waste time " + oneFT + " s.")

        var preFreSets = oneFreqSet.collect().map(x => x._1)

        for(round <- 2 to K if !preFreSets.isEmpty) {
            val ftk1 = System.nanoTime()
            // 生成频繁项的候选集
            val candidates = generateCandidates(preFreSets, round)
            // 将候选项集广播到各个分区
            val broadcastCandidates = sc.broadcast(candidates)

            //复杂度：len(transactions)*len(candidates)*round
            val curFreqSet = transations
                    .flatMap(x => verifyCandidates(x, broadcastCandidates.value))
                    .reduceByKey(_ + _)
                    .filter(x => x._2 >= freqThre)

            // 生成频繁round-Itemsets
            preFreSets = curFreqSet.collect().map(x => x._1)

            // 输出频繁项集并生成关联规则
            if (preFreSets.length > 0) {
                // 写入频繁round项集结果到hdfs
                curFreqSet.map(a => {
                    val out = a._1.mkString(",") + ":" + a._2.toString + "  " + (a._2 / numRecords).toString
                    out
                }).saveAsTextFile(output + "/" + infileName + "freqset-" + round)
                // 打印生成频繁round-Itemsets的时间
                val ftk2 = System.nanoTime()
                val FT = TimeUnit.SECONDS.convert(ftk2 - ftk1, TimeUnit.NANOSECONDS)
                println("generate "+ preFreSets.length + " Frequent  " + round + "-Item Set waste time " + FT + " s.")

                // 生成关联规则
                val asst1 = System.nanoTime()
                val broadcastCurFreqSet = sc.broadcast(preFreSets)
                // 生成所有可能的关联规则，然后筛选出置信度>=confidence的关联规则
                val associationRules = transations
                        .flatMap(x => verifyRules(x, broadcastCurFreqSet.value, round))
                        .reduceByKey{case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)}
                        .map(x => ((x._1._1, x._1._2), x._2._2 * 1.0 / x._2._1))
                        .filter(x => x._2 >= confidence)
                // 写入round-item的关联规则到hdfs
                associationRules.map(x => {
                    val out = x._1._1.mkString(",") + " -> " + x._1._2.mkString(",") + ":" + x._2.toString
                    out
                }).saveAsTextFile(output + "/" + infileName + "assrule-" + round)
                // 打印计算生成关联规则的时间
                val asst2 = System.nanoTime()
                val AT = TimeUnit.SECONDS.convert(asst2 - asst1, TimeUnit.NANOSECONDS)
                println("generate " + associationRules.collect().length + " association rules with "
                        + round + "-Item Set waste time " + AT + " s.")
            }
        }

        val endTime = System.nanoTime()
        val elapsedTIme = TimeUnit.SECONDS.convert(endTime - startTime, TimeUnit.NANOSECONDS)
        println("time used：" + elapsedTIme + " s")
        StdIn.readLine()
        sc.stop()
    }

    def verifyRules(transaction: HashSet[String], candidates: Array[HashSet[String]], curRound: Int): Array[((HashSet[String], HashSet[String]), (Int, Int))] = {
        for {
            set <- candidates

            i <- 1 until curRound

            iter = set.subsets(i).toArray
            l <- iter
            if (l.subsetOf(transaction))
            a = 1
            b = if (set.subsetOf(transaction)) 1 else 0
            r = set.diff(l)
        } yield ((l, r), (a, b))
    }

    def verifyCandidates(transation : HashSet[String], candidates: Array[HashSet[String]]): Array[(HashSet[String], Int)] = {
        for (c <- candidates if (c.subsetOf(transation))) yield (c, 1)
    }

    def generateCandidates(preFreSets : Array[HashSet[String]], curRound: Int): Array[HashSet[String]] = {
        // 复杂度：len(elements) * len(preFrestats)^2 * round^2
        val elements = preFreSets.reduce((a,b) => a.union(b))
        val canSets = preFreSets.flatMap( t => for (ele <- elements if(!t.contains(ele))) yield t.+(ele) ).distinct
        canSets.filter( set => {
            val iter = set.subsets(curRound - 1)
            var flag = true
            while (iter.hasNext && flag){
                flag = preFreSets.contains(iter.next())
            }
            flag
        })
    }
}