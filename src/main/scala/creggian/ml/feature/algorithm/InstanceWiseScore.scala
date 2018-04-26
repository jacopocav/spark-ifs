package creggian.ml.feature.algorithm

trait InstanceWiseScore extends Serializable {

    def getResult(matWithClass: Seq[Seq[Int]],
                  matWithFeatures: Seq[Seq[Seq[Int]]],
                  selectedVariablesSeq: Seq[Int],
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
