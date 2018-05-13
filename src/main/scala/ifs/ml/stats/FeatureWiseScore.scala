package ifs.ml.stats

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector


trait FeatureWiseScore extends Serializable {
    def apply(featureVector: Vector, labelRow: Vector, selectedVariables: Seq[LabeledPoint]): Double
}

object FeatureMRMR extends FeatureWiseScore {

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
}