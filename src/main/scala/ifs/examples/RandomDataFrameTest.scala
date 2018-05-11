package ifs.examples

import ifs.ml.feature.{FeatureSelector, FeatureSelectorModel, IterativeFeatureSelection}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
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

        val numSelectedFeatures = cliArgs.num_features()


        // Creating random matrix of single-digit integers
        val data = Vector.fill(cliArgs.rows(), cliArgs.cols())(Random.nextInt(10).toDouble)
        println(s"Created ${cliArgs.rows()}x${cliArgs.cols()} matrix with random 1-digit integers")

        //////////////////////////////////////////// Conventional Encoding ////////////////////////////////////////////
        var cResult: Option[Seq[Int]] = None

        if (cliArgs.conventional()) {
            println("\nStarting conventional encoding test...")
            val conventionalData = data.map(Row.fromSeq)

            // Naming features (+ label): <label, feat_1, feat_2, ...>
            val featureNames = "label" +: (1 until data(0).length).map("feat_" + _)

            // Conventional DataFrame construction
            val cRDD = spark.sparkContext.parallelize(conventionalData)
            val cSchema = StructType(featureNames map (col => StructField(col, DataTypes.DoubleType, nullable = false)))
            val cDF = spark.createDataFrame(cRDD, cSchema)

            println("Conventional DataFrame created")

            val featureColumns = featureNames.drop(1).toArray

            val va = new VectorAssembler()
                    .setInputCols(featureColumns)
                    .setOutputCol("features")

            val fs = new FeatureSelector()
                    .setFeaturesCol("features")
                    .setLabelCol("label")
                    .setOutputCol("selected")
                    .setNumSelectedFeatures(numSelectedFeatures)

            val pp = new Pipeline().setStages(Array(va, fs))

            print("Starting fit...")
            val res = doAndTime(pp.fit(cDF).stages(1).asInstanceOf[FeatureSelectorModel].selectedFeatures)
            println("done")

            cResult = Option(res._1)
            val cTime = res._2
            println(s"Conventional Encoding - Total time: $cTime ms")
        }


        //////////////////////////////////////////// Alternative Encoding /////////////////////////////////////////////
        var aResult: Option[Seq[(Int, Double)]] = None
        if (cliArgs.alternate()) {
            println("\nStarting alternate encoding test...")
            val alternativeData = data.transpose.drop(1).zip(data(0).indices).map {
                case (col, i) => LabeledPoint(i, Vectors.dense(col.toArray))
            }
            val labelVector = Vectors.dense(data.transpose.apply(0).toArray)

            val aRDD = spark.sparkContext.parallelize(alternativeData)

            print("Alternate RDD created\nStarting fit...")
            val res = doAndTime(IterativeFeatureSelection
                    .selectRows(aRDD, numSelectedFeatures, labelVector))
            println("done")


            aResult = Option(res._1)
            val aTime = res._2
            println(s"Alternate Encoding - Total time: $aTime ms")
        }


        if (cliArgs.conventional() & cliArgs.alternate()) {
            println()
            if (cResult.get == aResult.get.map(_._1))
                println(s"Selected features are the same:")
            else
                println(s"Selected features are different:")
            println(s"Conventional: ${cResult.get}")
            println(s"Alternate:    ${aResult.get.map(_._1)}")
        }

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
                "on it to select a given number of features (columns)\nOptions:")

    val cols: ScallopOption[Int] =
        opt[Int](name = "cols",
            short = 'c',
            validate = _ > 0,
            required = true,
            descr = "Number of columns")

    val rows: ScallopOption[Int] =
        opt[Int](name = "rows",
            short = 'r',
            validate = _ > 0,
            required = true,
            descr = "Number of rows")

    val num_features: ScallopOption[Int] =
        opt[Int](name = "num-features",
            short = 'n',
            required = true,
            descr = "Number of features (columns) to be selected")

    val alternate: ScallopOption[Boolean] =
        toggle(name = "alternate",
            default = Some(false),
            descrYes = "Tests feature selection using alternate (transposed) encoding (optional, default: disabled)")

    val conventional: ScallopOption[Boolean] =
        toggle(name = "conventional",
            default = Some(true),
            descrYes = "Tests feature selection using conventional encoding (default: enabled)")

    validate(alternate, conventional) { (a, c) =>
        if (!a & !c) Left("At least one between alternative and conventional flags must be active")
        else Right(Unit)
    }


    validate(cols, num_features) { (c, nf) =>
        if (nf > 0 & nf < c) Right(Unit)
        else Left(s"Number of selected features should be positive and lower than $c")
    }

    verify()
}
