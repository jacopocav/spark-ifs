package ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object SimpleApp {
    def main(args: Array[String]): Unit = {
        // Muting non-error messages
        Logger.getLogger("org").setLevel(Level.ERROR)
        Logger.getLogger("akka").setLevel(Level.ERROR)

        val logFile = "D:/Download/divina_commedia.txt" // Should be some file on your system
        val spark = SparkSession.builder
                .master("local[*]")
                .appName("Simple Application")
                .config("spark.local.dir", "D:\\spark-tmp")
                .getOrCreate

        import spark.implicits._

        val logData = spark.read.textFile(logFile)
        val words = logData.flatMap(_.toLowerCase.split("\\s+")).filter(!_.isEmpty)
        val wc = words.groupBy($"value").count

        println(s"Top 10 Most Common Words in Divine Comedy:")
        wc.orderBy($"count".desc).show(10)

        spark.stop()
    }
}
