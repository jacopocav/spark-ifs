package ifs.examples

import ifs.ml.feature.{FeatureSelector, FeatureSelectorModel, RowSelector, RowSelectorModel}
import ifs.util.extensions._
import ifs.util.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.rogach.scallop.{ScallopConf, ScallopOption, Subcommand}

import scala.io.Source

object CommandLine {
    def main(args: Array[String]): Unit = {
        val cli = new CsvArgs(args)

        if (cli.subcommand.contains(cli.gen)) {
            // CSV GENERATION MODE
            val subCli = cli.gen
            val rows = subCli.rows()
            val cols = subCli.cols()

            val altOption = subCli.altFile.map(af => (af, subCli.labels())).toOption

            randomMatricesToCsv(rows, cols, subCli.file.toOption, altOption)

            subCli.file foreach { file =>
                println(s"[gen] Conventional ${rows}x$cols matrix saved at $file")
            }

            altOption foreach { case (file, label) =>
                println(s"[gen] Alternate ${cols}x$rows matrix saved at $file")
                println(s"[gen] Associated label row saved at $label")
            }

        } else if (cli.subcommand.contains(cli.select)) {

            // CSV SELECTION MODE

            val subCli = cli.select
            implicit val verbose: Boolean = subCli.verbose()

            val spark: SparkSession = SparkSession.builder
                    .appName("spark-ifs")
                    .getOrCreate

            val cResult = subCli.file.toOption map { file =>
                val df = spark.read.option("inferSchema", "true")
                        .csv(file)
                        .withColumnRenamed("_c0", "label")

                verbosePrintln(s"[select-conv] DataFrame generated")

                val va = new VectorAssembler()
                        .setInputCols(df.columns.drop(1))
                        .setOutputCol("features")

                val fs = new FeatureSelector()
                        .setLabelCol("label")
                        .setFeaturesCol("features")
                        .setNumTopFeatures(subCli.num_features())
                        .setOutputCol("selected")

                val pp = new Pipeline().setStages(Array(va, fs))

                verbosePrint(s"[select-conv] Starting fit...")

                doAndTime {
                    val ppm = pp.fit(df)
                    ppm.transform(df)
                    verbosePrintln("done")
                    ppm.stages(1).asInstanceOf[FeatureSelectorModel].selectedFeatures
                }
            }

            val aResult = subCli.altFile.toOption map { file =>

                val df = spark.read.option("inferSchema", "true")
                        .csv(file)
                        .withColumn("id", monotonically_increasing_id())

                verbosePrintln("[select-alt] DataFrame generated")

                val labelRow = {
                    val labFile = Source.fromFile(subCli.labels())
                    val ret = labFile.bufferedReader.readLine.split(",").map(_.toDouble)
                    labFile.close
                    ret
                }

                verbosePrintln("[select-alt] Label row generated")

                val va = new VectorAssembler()
                        .setInputCols(df.columns.filterNot(_ == "id"))
                        .setOutputCol("features")

                val fs = new RowSelector()
                        .setIdCol("id")
                        .setLabelVector(labelRow)
                        .setFeaturesCol("features")
                        .setNumTopRows(subCli.num_features())
                        .setOutputCol("selected")
                        .setFiltered()

                val pp = new Pipeline().setStages(Array(va, fs))

                verbosePrint("[select-alt] Starting fit...")

                doAndTime {
                    val ppm = pp.fit(df)
                    ppm.transform(df)
                    verbosePrintln("done")
                    ppm.stages(1).asInstanceOf[RowSelectorModel].selectedRows
                }
            }

            println("/" * 35 + " RESULTS  " + "/" * 35)

            cResult foreach { case (selected, time) =>
                println(s"//// CONVENTIONAL ENCODING - Time: $time ms".paddedTo(76) + "////")
                verbosePrintln("//// SELECTED FEATURES:".paddedTo(76) + "////")
                verbosePrintln(s"//// [${selected.mkString(",")}]".paddedTo(76) + "////")
                verbosePrintln("/" * 80)
            }

            aResult foreach { case (selected, time) =>
                println(s"//// ALTERNATE ENCODING - Time: $time ms".paddedTo(76) + "////")
                verbosePrintln("//// SELECTED FEATURES:".paddedTo(76) + "////")
                verbosePrintln(s"//// [${selected.mkString(",")}]".paddedTo(76) + "////")
                verbosePrintln("/" * 80)

                cResult foreach { case (cSel, _) =>
                    if (selected == cSel) println("//// SELECTED FEATURES ARE IDENTICAL".paddedTo(76) + "////")
                }
            }

            println("/" * 80)

        } else cli.printHelp()

    }
}

class CsvArgs(val arguments: Seq[String]) extends ScallopConf(arguments) {
    banner("This program can be used to do IFS on datasets loaded from csv files " +
                "(and to generate random datasets to csv).")

    object gen extends Subcommand("gen") with FileArgs {
        banner("Generates a dataset with the given size.")
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
    }

    object select extends Subcommand("select") with FileArgs {
        banner("Selects the given number of features from the provided csv datasets. " +
            "NOTE: for this task spark-submit must be used.")
        val num_features: ScallopOption[Int] =
            opt[Int](name = "num-features",
                short = 'n',
                required = true,
                descr = "Number of features (columns) to be selected")
        val verbose: ScallopOption[Boolean] =
            toggle(name = "verbose",
                default = Some(false),
                descrYes = "Prints more information during execution",
                descrNo = "Only prints the results")
    }

    addSubcommand(gen)
    addSubcommand(select)

    verify()
}

trait FileArgs extends ScallopConf {
    val altFile: ScallopOption[String] =
        opt(name = "alt-file",
            descr = "Path to the csv in alternate encoding (without the label row)",
            default = None)

    val labels: ScallopOption[String] =
        opt(name = "labels",
            descr = "Path to the csv containing the label row (required for alternate encoding)",
            default = None)

    val file: ScallopOption[String] =
        opt(name = "file",
            descr = "Path to the csv in conventional encoding",
            default = None)

    codependent(altFile, labels)

    validateOpt(altFile, file) {
        case (None, None) => Left("At least one file path must be provided.")
        case _ => Right(Unit)
    }
}