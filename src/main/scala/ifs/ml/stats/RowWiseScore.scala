package ifs.ml.stats

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector

/**
  * This trait represents a function used to score a candidate feature in alternate encoding
  * (i.e. features are rows).
  */
trait RowWiseScore extends Serializable {
    /**
      * Computes a score for the candidate feature.
      * @param feature The candidate feature vector (i.e. row).
      * @param labelRow Vector containing label values associated to every instance (i.e. column).
      * @param selectedFeatures Collection of [[LabeledPoint]] that represent selected feature vectors
      *                          in the `features` field and their identifier in the `label` field.
      * @return The computed score
      */
    def apply(feature: Vector, labelRow: Vector, selectedFeatures: Seq[LabeledPoint]): Double

    /**
      * Expresses how scores should be compared (e.g: higher is better or vice versa).
      */
    def ordering: Ordering[Double]
}

/**
  * This object is used to compute the Minimum Redundancy Maximum Relevance (mRMR) score.
  */
object RowMRMR extends RowWiseScore {

    /**
      * Computes the mRMR score for the candidate feature.
      * @param feature The candidate feature vector (i.e. row).
      * @param labelRow Vector containing label values associated to every instance (i.e. column).
      * @param selectedFeatures Collection of [[LabeledPoint]] that represent selected feature vectors
      *                          in the `features` field and their identifier in the `label` field.
      * @return The computed score
      */
    def apply(feature: Vector, labelRow: Vector, selectedFeatures: Seq[LabeledPoint]): Double = {

        val labelScore = MutualInformation(feature, labelRow)

        val coefficient =
            if (selectedFeatures.nonEmpty) 1.0 / selectedFeatures.length
            else 0.0


        if (coefficient != 0.0) {

            // Summing all MI scores for every selected feature
            val featureScore = (0.0 /: selectedFeatures) ((acc, lp) => acc + MutualInformation(feature, lp.features))

            labelScore - (coefficient * featureScore)
        }
        else labelScore
    }

    /**
      * Descending score ordering (i.e. higher is better)
      */
    val ordering: Ordering[Double] = Ordering[Double].reverse
}