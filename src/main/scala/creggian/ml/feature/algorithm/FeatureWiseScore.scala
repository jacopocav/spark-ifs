package creggian.ml.feature.algorithm

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector

trait FeatureWiseScore extends Serializable {
    
    def getResult(featureVector: Vector,
                  classVector: Vector,
                  selectedVariablesArray: Array[LabeledPoint],
                  i: Int,
                  nfs: Int): Double
    
    def selectTop(i: Int, nfs: Int): Int = {
        1
    }
    
    def maxIterations(nfs: Int): Int = {
        nfs
    }
    
}	
