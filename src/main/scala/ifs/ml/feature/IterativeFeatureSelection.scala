package ifs.ml.feature

import breeze.linalg.Matrix
import ifs.ml.feature.algorithm.{FeatureMRMR, FeatureWiseScore, InstanceMRMR, InstanceWiseScore}
import org.apache.spark.SparkException
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD

import scala.collection.{immutable, mutable}

object IterativeFeatureSelection {

    private var _maxCategories = 10000

    def selectColumns(data: RDD[LabeledPoint],
                      num: Int,
                      score: InstanceWiseScore = InstanceMRMR): IndexedSeq[(Int, Double)] = {

        data.cache()
        val selectedFeatures = mutable.Buffer.empty[(Int, Double)]
        val numCols = data.first().features.size

        (1 to score.maxIterations(num)) foreach { _ =>

            val featureScores = computeScores(data, numCols, score, selectedFeatures.map(_._1))

            selectedFeatures += featureScores.maxBy(_._2)
        }
        // DEBUG
        println("\n============================= SELECTED FEATURES ============================= ")
        selectedFeatures.foreach(println)
        selectedFeatures.toVector
    }

    // Based on spark.mllib.stat.test.chiSquaredFeatures
    private def computeScores(data: RDD[LabeledPoint],
                              numCols: Int,
                              score: InstanceWiseScore = InstanceMRMR,
                              selectedFeatures: Seq[Int]): Seq[(Int, Double)] = {

        val selectedFeaturesB = data.context.broadcast(selectedFeatures)
        val candidates = (0 until numCols).filter(!selectedFeaturesB.value.contains(_))
        val results = mutable.Buffer.empty[(Int, Double)]
        var labels: Map[Double, Int] = null
        // at most 1000 columns at a time
        // TODO: remove batching
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

            if (labels == null)
            // Do this only once for the first column since labels are invariant across features.
            // labels is a map [label value -> label value index]
                labels = pairCounts.keys.filter(_._3 == -1).map(_._4).toSeq.distinct.zipWithIndex.toMap

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

                // TODO: try sparse matrices
                // Contingency matrices for every candidate -> selected feature pair
                val featureContingencies = mutable.LongMap.empty[Matrix[Long]]
                // Candidate -> label contingency matrix
                val labelContingency = Matrix.zeros[Long](numRows, labels.size)

                // Filling up contingency matrices
                keys foreach { case (_, feature1, col2, feature2) =>
                    val i = candidateIndices(feature1)
                    val j = if (col2 == -1) labels(feature2) else featureIndices(col2)(feature2)
                    if (col2 == -1)
                        labelContingency(i, j) += pairCounts(col, feature1, col2, feature2)
                    else {
                        if (!featureContingencies.contains(col2))
                        // TODO: try sparse matrix
                            featureContingencies(col2) = Matrix.zeros(numRows, featureDomains(col2))
                        featureContingencies(col2)(i, j) += pairCounts(col, feature1, col2, feature2)
                    }

                }
                results += (col -> score.getResult(labelContingency, featureContingencies.values.toSeq))
            }
            batch += 1
        }
        results.toVector
    }

    def maxCategories: Int = _maxCategories

    def maxCategories_=(value: Int): Unit = {
        require(value > 1, "Must have at least 1 category")
        _maxCategories = value
    }

    def rowWise(data: RDD[LabeledPoint], nfs: Int, labels: Vector, scorer: FeatureWiseScore = FeatureMRMR): Seq[(Int, Double)] = {

        val clVector_bc = data.context.broadcast(labels)

        var selectedScores = mutable.Buffer.empty[Double]
        var selectedFeatures = mutable.Buffer.empty[LabeledPoint]

        val selectedFeatureScores = mutable.Buffer.empty[(LabeledPoint, Double)]

        for (i <- 1 to scorer.maxIterations(nfs)) {

            val selectedVariable_bc = data.context.broadcast(selectedFeatures)
            val candidates = data.filter(x => !selectedVariable_bc.value.map(_.label).contains(x.label))

            val candidateScores = candidates.map(x => {
                val candidateScore = scorer.getResult(x.features, clVector_bc.value, selectedVariable_bc.value, i, nfs)
                (x.label, candidateScore)
            })

            val results = candidateScores.takeOrdered(scorer.selectTop(i, nfs))(Ordering[Double].reverse.on(_._2))

            for (ri <- results.indices) {
                val label = results(ri)._1
                val score = results(ri)._2

                val selectedFeature: LabeledPoint = {
                    val filteredDB = candidates.filter(x => x.label == label)
                    if (filteredDB.count() != 1) throw new RuntimeException("Multiple rows have the same identifier.")
                    else filteredDB.first()
                }

                selectedFeatureScores += ((selectedFeature, score))
                selectedScores += score
                selectedFeatures += selectedFeature
            }
        }

        selectedFeatureScores.map { case (lp, sc) => (lp.label.toInt, sc) }.toVector
    }
}
