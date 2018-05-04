package creggian.ml.feature.algorithm

import breeze.linalg.Matrix
import creggian.ml.feature.MutualInformation

trait InstanceWiseScore extends Serializable {

    def getResult(matWithClass: Seq[Seq[Int]],
                  matWithFeatures: Seq[Seq[Seq[Int]]],
                  selectedVariablesSeq: Seq[Int],
                  variableLevels: Seq[Double],
                  classLevels: Seq[Double],
                  i: Int,
                  nfs: Int): Double
    
    def selectTop(i: Int, nfs: Int): Int
    
    def maxIterations(nfs: Int): Int = nfs

    def getResult(labelContingency: Matrix[Long], featuresContingencies: Iterable[Matrix[Long]]): Double

}

object InstanceMRMR extends InstanceWiseScore {

    def getResult(matWithClass: Seq[Seq[Int]], matWithFeatures: Seq[Seq[Seq[Int]]], selectedVariablesIdx: Seq[Int], variableLevels: Seq[Double], classLevels: Seq[Double], i: Int, nfs: Int): Double = {
        mrmrMutualInformation(matWithClass, matWithFeatures, selectedVariablesIdx)
    }

    private def mrmrMutualInformation(matWithClass: Seq[Seq[Int]], matWithFeatures: Seq[Seq[Seq[Int]]], selectedVariablesIdx: Seq[Int]): Double = {

        val mrmrClass = MutualInformation.compute(matWithClass.map(_.map(_.toLong)))

        val mrmrFeatures = matWithFeatures.foldLeft(0.0)((a, f) => a + MutualInformation.compute(f.map(_.map(_.toLong))))

        val coefficient =
            if (selectedVariablesIdx.length > 1)
                1.0 / selectedVariablesIdx.length
            else
                1.0

        mrmrClass - (coefficient * mrmrFeatures)
    }

    override def selectTop(i: Int, nfs: Int): Int = 1

    def getResult(labelContingency: Matrix[Long], featuresContingencies: Iterable[Matrix[Long]]): Double = {
        val labelScore = MutualInformation.compute(labelContingency)

        val featuresScore = featuresContingencies.foldLeft(0.0) { (acc, mat) =>
            acc + MutualInformation.compute(mat)
        }

        val coefficient =
            if (featuresContingencies.nonEmpty) 1.0 / featuresContingencies.size
            else 0.0

        labelScore - (coefficient * featuresScore)
    }
}

/*object InstanceMaxRelevance extends InstanceWiseScore {

    def getResult(matWithClass: Seq[Seq[Int]], matWithFeatures: Seq[Seq[Seq[Int]]], selectedVariablesIdx: Seq[Int], variableLevels: Seq[Double], classLevels: Seq[Double], i: Int, nfs: Int): Double = {
        mrmrMutualInformation(matWithClass)
    }

    private def mrmrMutualInformation(matWithClass: Seq[Seq[Int]]): Double = {
        MutualInformation.compute(matWithClass)
    }

    override def selectTop(i: Int, nfs: Int): Int = nfs

}*/