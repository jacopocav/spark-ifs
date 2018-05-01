package ml

import creggian.ml.feature.FeatureSelector
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._

import scala.util.Random

object SimpleApp {
    def main(args: Array[String]): Unit = {
        // Muting non-error messages
        Logger.getLogger("org").setLevel(Level.ERROR)
        Logger.getLogger("akka").setLevel(Level.ERROR)

        val logFile = "D:/Download/divina_commedia.txt" // Should be some file on your system
        val spark: SparkSession = SparkSession.builder
                .master("local[*]")
                .appName("Simple Application")
                .config("spark.local.dir", "D:\\spark-tmp")
                .getOrCreate

        import spark.implicits._

        val testDS = Seq(
            (0, Vectors.dense(5.0, 0.0, Random.nextInt(10))),
            (1, Vectors.dense(4.0, 0.0, Random.nextInt(10))),
            (2, Vectors.dense(0.5, 4.0, Random.nextInt(10))),
            (3, Vectors.dense(0.5, 5.0, Random.nextInt(10)))
        ).toDF("label", "features").as[(Int, Vector)]

        val df = spark.read
                        .option("inferSchema", value = true)
                        .csv("D:/Download/mrmr.50x20.cw.c0.x1_8.csv")
                        .withColumnRenamed("_c0", "label")

        val featureColumns = df.columns.filter(_ != "label")

        val va = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features")

        val fs = new FeatureSelector()
                .setFeaturesCol("features")
                .setLabelCol("label")
                .setOutputCol("selected")
                .setNumSelectedFeatures(8)

        val pp = new Pipeline().setStages(Array(va, fs))

        val toDense = udf((v: Vector) => v.toDense)

        val outDF = pp.fit(df).transform(df)
                .withColumn("selectedDense", toDense($"selected"))

        println(s"Original dataframe:")
        df.show()

        println(s"Output dataframe:")
        outDF.select($"label", $"selectedDense").show(false)

        spark.stop()
    }
}
