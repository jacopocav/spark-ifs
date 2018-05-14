package ifs.ml.feature

import breeze.linalg.Matrix
import ifs.ml.stats.{ColumnMRMR, ColumnWiseScore, RowMRMR, RowWiseScore}
import org.apache.spark.SparkException
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD

import scala.collection.{immutable, mutable}

/**
  * Functions that perform Iterative Feature Selection on RDDs of LabeledPoints.
  */
object IterativeFeatureSelection {

    /**
      * Performs IFS on an RDD in conventional encoding (i.e. features are columns, instances are rows).
      * @param num Number of features to select (it should lower than the total number of features).
      * @param score Scoring function to use for selection.
      * @return Sequence of selected feature indices, paired to their score.
      */
    def selectColumns(data: RDD[LabeledPoint],
                      num: Int,
                      score: ColumnWiseScore = ColumnMRMR): Seq[(Int, Double)] = {

        data.cache()
        val selectedFeatures = mutable.Buffer.empty[(Int, Double)]
        val numCols = data.first().features.size
        val actualNum = if(num <= numCols) num else numCols

        (1 to actualNum) foreach { _ =>
            val featureScores = computeScores(data, numCols, score, selectedFeatures.map(_._1))

            // Takes the best-scored feature (according to the score's ordering)
            selectedFeatures += featureScores.sortBy(_._2)(score.ordering).head
        }
        // DEBUG
        //println("\n============================= SELECTED FEATURES ============================= ")
        //selectedFeatures.foreach(println)
        selectedFeatures.toVector
    }

    // Based on spark.mllib.stat.test.chiSquaredFeatures
    private def computeScores(data: RDD[LabeledPoint],
                              numCols: Int,
                              score: ColumnWiseScore = ColumnMRMR,
                              selectedFeatures: Seq[Int]): Seq[(Int, Double)] = {


        val selectedFeaturesB = data.context.broadcast(selectedFeatures)
        val candidates = (0 until numCols).filter(!selectedFeaturesB.value.contains(_))
        val results = mutable.Buffer.empty[(Int, Double)]
        var labels: Option[Map[Double, Int]] = None

        val maxCategories = 10000
        val batchSize = 1000
        var batch = 0
        while (batch * batchSize < numCols) {
            val startCol = batch * batchSize
            val endCol = startCol + math.min(batchSize, numCols - startCol)
            val candSubset = candidates.filter(c => c >= startCol && c < endCol)

            val pairCounts = data.mapPartitions { iter =>
                val distinctLabels = mutable.HashSet.empty[Double]
                val allDistinctFeatures =
                    immutable.LongMap((startCol.toLong until endCol).map(col => (col, mutable.HashSet.empty[Double])): _*)
                var i = 1

                iter.flatMap { case LabeledPoint(label, features) =>
                    if (i % batchSize == 0) {
                        if (distinctLabels.size > maxCategories)
                            throw new SparkException(s"Chi-square test expect factors (categorical values) but "
                                    + s"found more than $maxCategories distinct label values.")

                        allDistinctFeatures.foreach { case (col, distinctFeatures) =>
                            if (distinctFeatures.size > maxCategories)
                                throw new SparkException(s"Chi-square test expect factors (categorical values) but "
                                        + s"found more than $maxCategories distinct values in column $col.")
                        }
                    }
                    i += 1
                    distinctLabels += label

                    // Creates a tuple for every candidate-label and candidate-selected feature combination
                    // (candidate column, candidate value, other column (-1 for label), feature/label value)
                    candSubset.flatMap { col =>
                        allDistinctFeatures(col) += features(col)

                        selectedFeaturesB.value.map { col2 =>
                            (col, features(col), col2, features(col2))
                        } :+ (col, features(col), -1, label)
                    }
                }

            }.countByValue() // Count occurrences of every combination

            if (labels.isEmpty)
            // Do this only once for the first column since labels are invariant across features.
            // labels is a map [label value -> label value index]
                labels = Some(pairCounts.keys.filter(_._3 == -1).map(_._4).toSeq.distinct.zipWithIndex.toMap)

            pairCounts.keys.groupBy(_._1) foreach { case (col, keys) =>
                // Same as labels map, but for values of the candidate feature
                val candidateIndices = keys.map(_._2).toSeq.distinct.zipWithIndex.toMap
                val numRows = candidateIndices.size
                // Map containing the number of distinct values for every selected feature
                val featureDomains = keys.filter(_._3 != -1)
                        .groupBy(_._3)
                        .mapValues(_.map(_._4).toSeq.distinct.length)

                val featureIndices = keys.filter(_._3 != -1)
                        .groupBy(_._3)
                        .mapValues(_.map(_._4).toSeq.distinct.zipWithIndex.toMap)

                // Contingency matrices for every candidate -> selected feature pair
                val featureContingencies = mutable.LongMap.empty[Matrix[Long]]
                // Candidate -> label contingency matrix
                val labelContingency = Matrix.zeros[Long](numRows, labels.get.size)

                // Filling up contingency matrices
                keys foreach { case (_, feature1, col2, feature2) =>
                    val i = candidateIndices(feature1)
                    val j = if (col2 == -1) labels.get(feature2) else featureIndices(col2)(feature2)
                    if (col2 == -1)
                        labelContingency(i, j) += pairCounts(col, feature1, col2, feature2)
                    else {
                        if (!featureContingencies.contains(col2))
                            featureContingencies(col2) = Matrix.zeros(numRows, featureDomains(col2))
                        featureContingencies(col2)(i, j) += pairCounts(col, feature1, col2, feature2)
                    }

                }
                results += col -> score(labelContingency, featureContingencies.values.toSeq)
            }
            batch += 1
        }
        results.toVector
    }

    /**
      * Performs IFS on RDDs in alternate encoding (i.e. features are rows, instances are columns).
      * @param data RDD of [[LabeledPoint]] where the label must be a unique identifier.
      * @param num Number of features to select (must be lower or equal to total number of features).
      * @param labelsRow Vector containing label values associated to every instance (i.e. column).
      * @param score The scoring function used for selection.
      * @return Sequence of selected feature identifiers paired to their score.
      */
    def selectRows(data: RDD[LabeledPoint],
                   num: Int, labelsRow: Vector,
                   score: RowWiseScore = RowMRMR): Seq[(Double, Double)] = {

        val rowSize = data.count()
        val actualNum = if(num <= rowSize) num else rowSize

        val labelsRowB = data.context.broadcast(labelsRow)

        val selectedFeatureScores = mutable.Buffer.empty[(LabeledPoint, Double)]

        (1L to actualNum) foreach { _ =>
            val selectedFeaturesB = data.context.broadcast(selectedFeatureScores.map(_._1).toVector)
            val candidates = data.filter(x => !selectedFeaturesB.value.map(_.label).contains(x.label))

            val candidateScores = candidates map {c =>
                val cScore = score(c.features, labelsRowB.value, selectedFeaturesB.value)
                (c.label, cScore)
            }

            // Takes the best-scoring feature
            val (id, bestScore) = candidateScores.takeOrdered(1)(score.ordering on (_._2))(0)

            val selectedFeature: LabeledPoint = {
                val filteredDB = candidates.filter(_.label == id)
                if (filteredDB.count() != 1) sys.error(s"Multiple rows have the same identifier ($id).")
                else filteredDB.first()
            }

            selectedFeatureScores += selectedFeature -> bestScore

        }

        selectedFeatureScores.map{ case (lp, sc) => (lp.label, sc) }.toVector
    }
}
