package ifs.ml.stats

import breeze.linalg.Matrix

/**
  * This trait represents a function used to score a candidate feature in conventional encoding
  * (i.e. features are columns).
  */
trait ColumnWiseScore extends Serializable {
    /**
      * Computes a score for the candidate feature by using contingency matrices.
      * @param labelContingency The label-candidate contingency matrix.
      * @param featuresContingencies All candidate-selected feature contingency matrices.
      * @return A decimal score.
      */
    def apply(labelContingency: Matrix[Long], featuresContingencies: Seq[Matrix[Long]]): Double

    /**
      * Expresses how scores should be compared (e.g: higher is better or vice versa).
      */
    def ordering: Ordering[Double]
}

/**
  * This object is used to compute the Minimum Redundancy Maximum Relevance (mRMR) score.
  */
object ColumnMRMR extends ColumnWiseScore {
    /**
      * Computes the mRMR score of the candidate feature.
      * @param labelContingency The label-candidate contingency matrix.
      * @param featuresContingencies All candidate-selected feature contingency matrices.
      * @return A decimal score (higher is better).
      */
    def apply(labelContingency: Matrix[Long], featuresContingencies: Seq[Matrix[Long]]): Double = {
        val labelScore = MutualInformation(labelContingency)

        val coefficient =
            if (featuresContingencies.nonEmpty) 1.0 / featuresContingencies.length
            else 0.0

        if(coefficient != 0.0) {
            val featuresScore = (0.0 /: featuresContingencies)((acc, mat) => acc + MutualInformation(mat))
            labelScore - (coefficient * featuresScore)
        } else labelScore
    }

    /**
      * Descending score ordering (higher is better)
      */
    val ordering: Ordering[Double] = Ordering[Double].reverse
}