package ifs.ml.stats

import breeze.linalg.Matrix

trait InstanceWiseScore extends Serializable {
    def apply(labelContingency: Matrix[Long], featuresContingencies: Seq[Matrix[Long]]): Double
}

object InstanceMRMR extends InstanceWiseScore {
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
}