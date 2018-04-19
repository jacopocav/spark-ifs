package creggian.ml.feature.algorithm

abstract class InstanceWiseAbstractScore extends Serializable {
    
    def getResult(matWithClass: Seq[Seq[Long]],
                  matWithFeatures: Seq[Seq[Seq[Long]]],
                  selectedVariablesSeq: Seq[Long],
                  variableLevels: Seq[Double],
                  classLevels: Seq[Double],
                  i: Int,
                  nfs: Int): Double
    
    def selectTop(i: Int, nfs: Int): Int = {
        1
    }
    
    def maxIterations(nfs: Int): Int = {
        nfs
    }
    
}
