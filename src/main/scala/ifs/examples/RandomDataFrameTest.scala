package ifs.examples

import ifs.ml.feature.{FeatureSelector, FeatureSelectorModel}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.rogach.scallop.{ScallopConf, ScallopOption}

import scala.util.Random

object RandomDataFrameTest {
    def main(args: Array[String]): Unit = {
        // Muting non-error messages
        Logger.getLogger("org").setLevel(Level.ERROR)
        Logger.getLogger("akka").setLevel(Level.ERROR)

        val cliArgs = new CliArguments(args)

        val spark: SparkSession = SparkSession.builder
                .master("local[*]")
                .appName("Simple Application")
                .getOrCreate

        // 1000 rows with 50 columns filled with random integers in range [0,9]
        val data = IndexedSeq.fill(cliArgs.rows(), cliArgs.cols())(Random.nextInt(10).toDouble)
                .map(row => Row.fromSeq(row))

        // Naming columns: <label, col_1, col_2, ...>
        val colNames = "label" +: (1 until cliArgs.cols()).map("col_" + _)

        // Creating the DataFrame
        val newRdd = spark.sparkContext.parallelize(data)
        val schema = StructType(colNames map (col => StructField(col, DataTypes.DoubleType, nullable = false)))
        val df = spark.createDataFrame(newRdd, schema)

        val featureColumns = colNames.drop(1).toArray

        val numSelectedFeatures = 10

        val va = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features")

        val fs = new FeatureSelector()
                .setFeaturesCol("features")
                .setLabelCol("label")
                .setOutputCol("selected")
                .setNumSelectedFeatures(numSelectedFeatures)

        val pp = new Pipeline().setStages(Array(va, fs))

        val (ppmod, newTime) = doAndTime(pp.fit(df).stages(1).asInstanceOf[FeatureSelectorModel].selectedFeatures)
        println(s"Total time: $newTime ms")

        spark.stop()
    }

    def doAndTime[T](block: => T): (T, Long) = {
        val start = System.nanoTime()
        val ret = block
        val end = System.nanoTime()
        (ret, (end - start) / 1000000)
    }
}

class CliArguments(arguments: Seq[String]) extends ScallopConf(arguments) {
    banner(
        "This program generates a random DataFrame of the specified size and does IFS " +
                "on it to select a given number of features (columns)")

    val cols: ScallopOption[Int] =
        opt[Int](validate = _ > 0,
            required = true,
            descr = "Number of columns")

    val rows: ScallopOption[Int] =
        opt[Int](validate = _ > 0,
            required = true,
            descr = "Number of rows")

    val num_features: ScallopOption[Int] =
        opt[Int](required = true,
            descr = "Number of features (columns) to be selected")

    validate(cols, num_features) { (c, nf) =>
        if (nf > 0 & nf < c) Right(Unit)
        else Left(s"Number of selected features should be positive and lower than $c")
    }
    verify()
}
