package main.scala.Apriori.standalone

import java.io.PrintWriter
import java.util.concurrent.TimeUnit
import scala.collection.mutable
import scala.collection.mutable.HashSet
import scala.io._
import scala.util.control.Breaks._

object Apriori {
    def main(args: Array[String]): Unit = {
        val startTime = System.nanoTime()
        System.out.println(args)
        if(args.length < 4){
            System.err.println("Usage: <inputDataset> <outputDir>")
            System.exit(-1)
        }
        val input = args(0)//"data/simpleTest.txt"
        val output = args(1) //if(args(1).endsWith("/")) args(1).substring(0, args(1).length() - 1) else args(1) // "data/simpleTest/"
        val support = args(2).toDouble // 0.5
        val K = args(3).toInt
        val infileName = input.substring(input.lastIndexOf('/')+1, input.lastIndexOf('.'))

        val data = Source.fromFile(input, "UTF-8").getLines().toArray

        val ft0 = System.nanoTime()
        val transactions = data.map(elem => {
            val split = elem.split("\\s+")
            val tmp = new HashSet[String]
            for (i <- 0 until split.length) {
                tmp.add(split(i))
                tmp
            }
            tmp
        })

        val numRecords = transactions.length.toDouble
        val freqThre = numRecords * support

        println("frequency threshold：" + freqThre)
        val one = transactions.flatMap(_.seq)
        val oneMap = new mutable.HashMap[String, Int]
        one.map(elem => {
            if (oneMap.contains(elem)) {
            val value = oneMap(elem)
            oneMap += (elem -> (value+1))
            } else {
            oneMap += (elem -> 1)
            }
        })
        val oneFreqSet = oneMap.toArray
            .filter(x => x._2 >= freqThre)
            .map(x => (HashSet(x._1), x._2))

        val ft1 = System.nanoTime()
        val oneFT = TimeUnit.SECONDS.convert(ft1 - ft0, TimeUnit.NANOSECONDS)
        println("generate " + oneFreqSet.length + " Frequent 1-Item Set waste time " + oneFT + " s.")
        val ps1 = new PrintWriter((output + "/" + infileName + "result-1.txt"))
        for (elem <- oneFreqSet)
            ps1.println(elem._1.mkString(" ") + " : " + elem._2.toString)
        ps1.close()

        var preFreSets: Array[HashSet[String]] = oneFreqSet.map(x => x._1)
        breakable {
            for (round <- 2 to K if !preFreSets.isEmpty) {
                val ftk1 = System.nanoTime()
                val candidates = generateCandidates(preFreSets, round)
                val curSet = transactions
                    .flatMap(x => verifyCandidates(x, candidates))
                val curMap = new mutable.HashMap[HashSet[String], Int]
                curSet.map(elem => {
                if (curMap.contains(elem)) {
                    val value = curMap(elem)
                    curMap += (elem -> (value + 1))
                } else {
                    curMap += (elem -> 1)
                }
                })
                val curFreqSet = curMap.toArray
                    .filter(x => x._2 >= freqThre)
                if (curFreqSet.length == 0) {
                    break
                }
                val ftk2 = System.nanoTime()
                val FT = TimeUnit.SECONDS.convert(ftk2 - ftk1, TimeUnit.NANOSECONDS)
                println("generate "+ curFreqSet.length + " Frequent  " + round + "-Item Set waste time " + FT + " s.")

                val ps2 = new PrintWriter((output + "/" + infileName + "result-" + round + ".txt"))
                for (elem <- curFreqSet) {
                    ps2.println(elem._1.mkString(" ") + " : " + elem._2)
                }
                ps2.close()
                preFreSets = curFreqSet.map(x => x._1)
            }
        }
        val endTime = System.nanoTime()
        val elapsedTIme = TimeUnit.SECONDS.convert(endTime - startTime, TimeUnit.NANOSECONDS)
        println("time used：" + elapsedTIme + " s")
    }

    def verifyCandidates(transation : HashSet[String], candidates: Array[HashSet[String]]): Array[HashSet[String]] = {
        for (c <- candidates if (c.subsetOf(transation))) yield (c)
    }

    def generateCandidates(preFreSets : Array[HashSet[String]], curRound: Int): Array[HashSet[String]] = {
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

    def Divide(a : Int, b: Double) : Double = {
        (a * 1.0) / b
    }
}
