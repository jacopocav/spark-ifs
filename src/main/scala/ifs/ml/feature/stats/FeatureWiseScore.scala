package ifs.ml.feature.stats

import breeze.linalg.Matrix
import ifs.util.extensions._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector


trait FeatureWiseScore extends Serializable {
    def apply(featureVector: Vector, classVector: Vector, selectedVariables: Seq[LabeledPoint]): Double
}

object FeatureMRMR extends FeatureWiseScore {

    def apply(feature: Vector, classVector: Vector, selectedFeatures: Seq[LabeledPoint]): Double = {

        val labelScore = discreteMI(feature, classVector)

        val coefficient =
            if (selectedFeatures.nonEmpty) 1.0 / selectedFeatures.length
            else 0.0


        if (coefficient != 0.0) {

            // Summing all MI scores for every selected feature
            val featureScore = (0.0 /: selectedFeatures) ((acc, lp) => acc + discreteMI(feature, lp.features))

            labelScore - (coefficient * featureScore)
        }
        else labelScore
    }

    private def discreteMI(a: Vector, b: Vector): Double = {

        require(a.size == b.size, s"Vectors a and b must have the same length: a:${a.size} != b:${b.size}")

        val aLevels = a.toIndexMap
        val bLevels = b.toIndexMap

        val mat = Matrix.zeros[Long](bLevels.size, aLevels.size)

        var nonZeroCount = 0

        // Counting all occurrences (av, bv) where av is not zero
        // Note: all inactive elements of a Vector are zero, but there may be some active zeros (e.g: in dense form)
        a.nonZeros foreach { case (index, av) =>
            val bv = b(index)
            mat(bLevels(bv), aLevels(av)) += 1
            nonZeroCount += 1
        }

        // Counting all remaining occurrences that are not (0,0)
        // All the remaining occurrences are of type (bv, 0)
        b.nonZeros foreach { case (index, bv) =>
            val av = a(index)
            if (av == 0.0) {
                // If av != 0 it's already been counted in the previous loop
                mat(bLevels(bv), aLevels(av)) += 1
                nonZeroCount += 1
            }
        }

        // Counting all (0,0) occurrences
        if (aLevels.contains(0.0) & bLevels.contains(0.0))
            mat(bLevels(0.0), aLevels(0.0)) = a.size - nonZeroCount

        MutualInformation.compute(mat)
    }
}