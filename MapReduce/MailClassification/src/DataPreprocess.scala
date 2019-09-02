import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.FileReader
import java.io.FileWriter
import java.io.IOException
import java.io.File
import java.util.concurrent.TimeUnit

object DataPreprocess {
    val stopwordsPath = "input/stopwords_en.txt"
    val corpusPath = "input/20_newsgroup"
    val outputPath = "output/MailCollection"

    def main(args:Array[String]): Unit = {
        val startTime = System.nanoTime()
        val es = new EnglishSegment(stopwordsPath)
        try {
            val writer = new BufferedWriter(new FileWriter(outputPath))
            val corpus = new File(corpusPath).listFiles()
            for (subdir <- 0 until corpus.length) {
                val mailLists = corpus(subdir).listFiles()
                for (mail <- 0 until mailLists.length) {
                    val sb = new StringBuilder
                    val reader = new BufferedReader(new FileReader(mailLists(mail)))
                    var line:String = null
                    while ({line = reader.readLine(); line != null}) {
                        sb.append(line)
                    }
                    reader.close()
                    val terms = es.segment(sb.toString())
                    writer.write(corpus(subdir).getName() + "\t" + terms + "\n")
                }
            }
            writer.close();
        } catch {
            case e: IOException => e.printStackTrace()
        }
        val endTime = System.nanoTime()
        val timestamp = TimeUnit.SECONDS.convert(endTime - startTime, TimeUnit.NANOSECONDS)
        println("preprocess used " + timestamp + " s.")
    }
}
