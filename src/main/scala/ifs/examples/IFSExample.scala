package ifs.examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

import scala.util.Random

object IFSExample {
    def main(args: Array[String]): Unit = {
        // Muting non-error messages
        Logger.getLogger("org").setLevel(Level.ERROR)
        Logger.getLogger("akka").setLevel(Level.ERROR)

        val cliArgs = new CliArguments(args)

        val spark: SparkSession = SparkSession.builder
                .master("local[*]")
                .appName("Simple Application")
                .getOrCreate

        import spark.implicits._

        val data = IndexedSeq.fill(cliArgs.rows(), cliArgs.cols())(Random.nextInt(10).toDouble)
                .map(row => Row.fromSeq(row))

        // Naming columns: <label, col_1, col_2, ...>
        val colNames = "label" +: (1 until cliArgs.cols()).map("col_" + _)
        val columns = colNames.map(col)

        // Creating the DataFrame
        val newRdd = spark.sparkContext.parallelize(data)
        val schema = StructType(colNames map (col => StructField(col, DataTypes.DoubleType, nullable = false)))

        val df = spark.createDataFrame(newRdd, schema).withColumn("id", monotonically_increasing_id())

        val labelCol = df.select($"label").map(row => row(0).asInstanceOf[Double]).collect().toVector

        val pivdf = df.drop($"label")
                .groupBy(columns.drop(1): _*)
                .pivot("id")
                .agg(first(columns(1)), columns.drop(2).map(first): _*)

        pivdf.drop(colNames.drop(1): _*).show(false)
    }
}
