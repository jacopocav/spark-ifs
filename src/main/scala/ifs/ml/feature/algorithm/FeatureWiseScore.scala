package ifs.ml.feature.algorithm

import ifs.ml.feature.MutualInformation
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

trait FeatureWiseScore extends Serializable {

    def getResult(featureVector: Vector, classVector: Vector, selectedVariablesArray: Seq[LabeledPoint], i: Int, nfs: Int): Double
    
    def selectTop(i: Int, nfs: Int): Int = 1
    
    def maxIterations(nfs: Int): Int = nfs
    
}

object FeatureMRMR extends FeatureWiseScore {

    def getResult(featureVector: Vector, classVector: Vector, selectedVariablesArray: Seq[LabeledPoint], i: Int, nfs: Int): Double = {
        this.mrmrMutualInformation(featureVector, classVector, selectedVariablesArray)
    }

    private def mrmrMutualInformation(featureVector: Vector, classVector: Vector, selectedVariablesArray: Seq[LabeledPoint]) = {

        val classLevels = classVector match {
            case v: SparseVector => v.values.distinct ++ Array(0.0)
            case v: DenseVector => v.values.distinct
        }
        val variableLevelsAll = for (vector <- Array(featureVector) ++ selectedVariablesArray.map(x => x.features)) yield {
            vector match {
                case v: SparseVector => v.values.distinct ++ Array(0.0)
                case v: DenseVector => v.values.distinct
            }
        }
        val variableLevels = variableLevelsAll.reduce(_ ++ _).distinct

        val mrmrClass = mutualInformationDiscrete(featureVector, classVector, variableLevels, classLevels)

        var mrmrFeatures = 0.0
        for (j <- selectedVariablesArray.indices) {
            val sLP = selectedVariablesArray(j)
            mrmrFeatures = mrmrFeatures + mutualInformationDiscrete(featureVector, sLP.features, variableLevels, variableLevels)
        }

        var coefficient = 1.0
        if (selectedVariablesArray.length > 1) coefficient = 1.0 / selectedVariablesArray.length.toDouble

        mrmrClass - (coefficient * mrmrFeatures)
    }

    private def mutualInformationDiscrete(a: Vector, b: Vector, aLevels: Array[Double], bLevels: Array[Double]): Double = {

        if (a.size != b.size) throw new RuntimeException("Vectors 'a' and 'b' must have the same length: a = " + a.size + " - b = " + b.size)

        val mat = Array.fill[Long](bLevels.length, aLevels.length)(0)

        val aIdx = a match {
            case v: SparseVector => v.indices.toIndexedSeq
            case v: DenseVector  => 0 until v.size
        }
        val bIdx = b match {
            case v: SparseVector => v.indices.toIndexedSeq
            case v: DenseVector  => 0 until v.size
        }

        var counter = 0L
        for (i <- aIdx) {
            val aValue = a(i) // real value
            val bValue = if (bIdx.contains(i)) b(i) else 0.0// real value

            val aValueIdxMat = aLevels.indexWhere(_ == aValue)
            val bValueIdxMat = bLevels.indexWhere(_ == bValue)

            mat(bValueIdxMat)(aValueIdxMat) = mat(bValueIdxMat)(aValueIdxMat) + 1L
            counter += 1L
        }

        for (i <- bIdx) {
            if (!aIdx.contains(i)) {
                val aValue = 0.0 // real value
                val bValue = b(i) // real value

                val aValueIdxMat = aLevels.indexWhere(_ == aValue)
                val bValueIdxMat = bLevels.indexWhere(_ == bValue)

                mat(bValueIdxMat)(aValueIdxMat) = mat(bValueIdxMat)(aValueIdxMat) + 1L
                counter += 1L
            }
        }

        val aZeroIdxMat = aLevels.indexWhere(_ == 0.0)
        val bZeroIdxMat = bLevels.indexWhere(_ == 0.0)
        if (aZeroIdxMat >= 0 && bZeroIdxMat >= 0) mat(bZeroIdxMat)(aZeroIdxMat) = a.size - counter

        MutualInformation.compute(mat)
    }
}