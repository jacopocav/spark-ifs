package creggian.ml.feature

import creggian.collection.mutable._
import creggian.ml.feature.algorithm.{FeatureMRMR, FeatureWiseScore, InstanceMRMR, InstanceWiseScore}
import org.apache.spark.SparkException
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.immutable.{Map, Vector => ScalaVector}
import scala.collection.mutable
import scala.collection.mutable.{Map => MMap}

object IterativeFeatureSelection {

    val maxCategories = 10000

    def columnWise(data: RDD[LabeledPoint], nfs: Int, score: InstanceWiseScore = InstanceMRMR): ScalaVector[(Int, Double)] = {
        val labelCounts = data.map(x => x.label).countByValue()
        val selectedFeatureCounts = MMap[Int, Map[Double, Long]]()

        var colNum: Option[Int] = None
        val labelDomain = labelCounts.keys.toArray
        val featuresDomain = data.map({
            case LabeledPoint(_, v: Vector) =>
                if (colNum.isEmpty) colNum = Some(v.size)
                v.toArray.toSet

        }).reduce(_ union _) + 0.0

        val labelCounts_bc = data.context.broadcast(labelCounts)
        val labelDomain_bc = data.context.broadcast(labelDomain)
        val featuresDomain_bc = data.context.broadcast(featuresDomain.toVector)

        val candidateIdx = IndexArray.dense(colNum.get).fill()
        val selectedIdx = IndexArray.sparse(colNum.get)
        var selectedIdxScore = Seq.empty[Double]

        for (i <- 1 to score.maxIterations(nfs)) {

            val candidateIdx_bc = data.context.broadcast(candidateIdx)
            val selectedIdx_bc = data.context.broadcast(selectedIdx)
            val selectedFeatureCounts_bc = data.context.broadcast(selectedFeatureCounts)

            val contingencyTableItems = data.mapPartitions(it => {
                val i_c = candidateIdx_bc.value
                val i_s = selectedIdx_bc.value
                val d_l = labelDomain_bc.value
                val d_f = featuresDomain_bc.value

                // key = ((candidate index, -1, candidate value index, candidate label value index), count)
                val labelMatrix = MMap[(Int, Int, Int, Int), Int]()
                //key = ((candidate index, selected feat. index, candidate value index, selected feat. value index), count)
                val featureMatrix = MMap[(Int, Int, Int, Int), Int]()

                it foreach { p =>
                    val features = p.features

                    features foreachActive { (fi, fv) =>

                        if (i_c.contains(fi) && fv != 0.0) {
                            val labelIndex = d_l.indexOf(p.label)
                            val valIndex = d_f.indexOf(fv)

                            val lkey = (fi, -1, valIndex, labelIndex) // -1 is a placeholder for the label column index

                            labelMatrix(lkey) = labelMatrix.getOrElse(lkey, 0) + 1

                            for (sfi <- i_s.indices) {
                                val selectedValIndex = d_f.indexOf(features(sfi))

                                val fkey = (fi, sfi, valIndex, selectedValIndex)

                                featureMatrix(fkey) = featureMatrix.getOrElse(fkey, 0) + 1
                            }
                        }
                    }
                }
                (labelMatrix.toSeq ++ featureMatrix.toSeq).iterator
            })

            val candidateScores = contingencyTableItems.reduceByKey(_ + _).map {
                case ((ci, sfi, vi, svi), c) => (ci, (sfi, vi, svi, c))
            }.groupByKey.map { entry =>

                val (key, buffer) = entry

                val cvt = labelCounts_bc.value
                val fvt = selectedFeatureCounts_bc.value
                val cl = labelDomain_bc.value
                val fl = featuresDomain_bc.value
                val cln = cl.length
                val fln = fl.length

                // create the intermediate data structures for discrete features
                val matWithClass = Seq.fill[Int](cln, fln)(0)
                val matWithFeaturesMap = MMap[Int, ScalaVector[ScalaVector[Int]]]()

                // fill in the data from the Map step into the intermediate data structures
                val iter = buffer.iterator
                while (iter.hasNext) {
                    val item = iter.next
                    val (otherIdx, fcMatIdx, otherMatIdx, value) = item

                    if (otherIdx == -1) {
                        matWithClass(otherMatIdx)(fcMatIdx) = value
                    } else {
                        if (matWithFeaturesMap.contains(otherIdx)) matWithFeaturesMap(otherIdx)(otherMatIdx)(fcMatIdx) = value
                        else {
                            matWithFeaturesMap(otherIdx) = ScalaVector.fill[Int](fln, fln)(0)
                            matWithFeaturesMap(otherIdx)(otherMatIdx)(fcMatIdx) = value
                        }
                    }
                }

                // to save bandwidth, we did not produced tuples where the candidate
                // feature's value id zero. This is particularly useful in sparse datasets
                // but it can save also some space in dense datasets.
                val fcIdxCtZero = fl.indexOf(0.0)
                val marginalSum = matWithClass.map(_.sum)
                for (i <- 0 until cln) {
                    val totRowI = if (cvt.contains(cl(i))) cvt(cl(i)) else 0L
                    if (totRowI - marginalSum(i) < 0) throw new RuntimeException("Marginal sum greater than total")
                    matWithClass(i)(fcIdxCtZero) = totRowI - marginalSum(i)
                }
                for (fsIdx <- matWithFeaturesMap.keys) {
                    val ct = matWithFeaturesMap(fsIdx)
                    val marginalSum = ct.map(_.sum)
                    for (i <- 0 until fln) {
                        // matWithFeaturesMap(key) represents the 'fln' x 'fln' contingency table of the selected feature 'fsIdx' (rows) with the candidate feature (cols)
                        // 'i' is the index of all the possible values of 'fl' (of length fln) => row index
                        // 'fcIdxCtZero' is the column index where the feature candidate value is zero
                        val fsCt = fvt(fsIdx)
                        val totRowI = if (fsCt.contains(fl(i))) fvt(fsIdx)(fl(i)) else 0L
                        if (totRowI - marginalSum(i) < 0) throw new RuntimeException("Marginal sum greater than total")
                        matWithFeaturesMap(fsIdx)(i)(fcIdxCtZero) = totRowI - marginalSum(i)
                    }
                }

                // create the missing data structured needed for the interface score class
                val selectedFeaturesIdx = matWithFeaturesMap.keys.toSeq
                val matWithFeatures = matWithFeaturesMap.values.toSeq

                val candidateScore = score.getResult(matWithClass, matWithFeatures, selectedFeaturesIdx, fl, cl, i, nfs)

                (key, candidateScore)
            }

            val results = candidateScores.takeOrdered(score.selectTop(i, nfs))(Ordering[Double].reverse.on(_._2))

            for (ri <- results.indices) {
                // Example of step3 output (key, score) = (2818,0.8462824341015066)
                val novelSelectedFeatureIdx = results(ri)._1
                val novelSelectedFeatureScore = results(ri)._2

                selectedIdx.add(novelSelectedFeatureIdx)
                selectedIdxScore = selectedIdxScore :+ novelSelectedFeatureScore
                selectedFeatureCounts(novelSelectedFeatureIdx) = data.map(x => x.features(novelSelectedFeatureIdx)).countByValue().toMap
                candidateIdx.remove(novelSelectedFeatureIdx)
            }
        }

        selectedIdx.indices.zip(selectedIdxScore).toVector
    }

    def rowWise(data: RDD[LabeledPoint], nfs: Int, classVector: Vector, score: FeatureWiseScore = FeatureMRMR): Array[(Int, Double)] = {

        val clVector_bc = data.context.broadcast(classVector)

        var selectedVariableScore = Array[Double]()
        var selectedVariable = Array[LabeledPoint]()

        for (i <- 1 to score.maxIterations(nfs)) {

            val selectedVariable_bc = data.context.broadcast(selectedVariable)
            val varrCandidate = data.filter(x => !selectedVariable_bc.value.map(lp => lp.label).contains(x.label))

            val candidateScores = varrCandidate.map(x => {
                val candidateScore = score.getResult(x.features, clVector_bc.value, selectedVariable_bc.value, i, nfs)
                (x.label, candidateScore)
            })

            val results = candidateScores.takeOrdered(score.selectTop(i, nfs))(Ordering[Double].reverse.on(_._2))

            for (ri <- results.indices) {
                val novelSelectedFeatureLabel = results(ri)._1
                val novelSelectedFeatureScore = results(ri)._2

                val newSelectedLabel_bc = data.context.broadcast(novelSelectedFeatureLabel)
                val newSelectedLPAll = varrCandidate.filter(x => x.label == newSelectedLabel_bc.value)

                if (newSelectedLPAll.count() != 1L) throw new RuntimeException("Error: Multiple features with the same ID")

                val newSelectedLP = newSelectedLPAll.first()

                selectedVariableScore = selectedVariableScore ++ Array(novelSelectedFeatureScore)
                selectedVariable = selectedVariable ++ Array(newSelectedLP)
            }
        }

        selectedVariable.map(x => x.label.toInt).zip(selectedVariableScore)
    }

    def compress(rdd: RDD[LabeledPoint], selectedIdx: Array[Int]): RDD[LabeledPoint] = {
        rdd.map(x => {
            val label = x.label
            val features = Vectors.dense(selectedIdx.map(idx => x.features(idx)))
            LabeledPoint(label, features.compressed)
        })
    }

    def chiSquaredFeatures(data: RDD[LabeledPoint],
                           score: InstanceWiseScore = InstanceMRMR): Array[Any] = {
        val numCols = data.first().features.size
        val results = new Array[Any](numCols)
        var labels: Map[Double, Int] = null
        // at most 1000 columns at a time
        val batchSize = 1000
        var batch = 0
        while (batch * batchSize < numCols) {
            // The following block of code can be cleaned up and made public as
            // chiSquared(data: RDD[(V1, V2)])
            val startCol = batch * batchSize
            val endCol = startCol + math.min(batchSize, numCols - startCol)
            val pairCounts = data.mapPartitions { iter =>
                val distinctLabels = mutable.HashSet.empty[Double]
                val allDistinctFeatures: Map[Int, mutable.HashSet[Double]] =
                    Map((startCol until endCol).map(col => (col, mutable.HashSet.empty[Double])): _*)
                var i = 1
                iter.flatMap { case LabeledPoint(label, features) =>
                    if (i % batchSize == 0) {
                        if (distinctLabels.size > maxCategories) {
                            throw new SparkException(s"Chi-square test expect factors (categorical values) but "
                                    + s"found more than $maxCategories distinct label values.")
                        }
                        allDistinctFeatures.foreach { case (col, distinctFeatures) =>
                            if (distinctFeatures.size > maxCategories) {
                                throw new SparkException(s"Chi-square test expect factors (categorical values) but "
                                        + s"found more than $maxCategories distinct values in column $col.")
                            }
                        }
                    }
                    i += 1
                    distinctLabels += label
                    (startCol until endCol).map { col =>
                        val feature = features(col)
                        allDistinctFeatures(col) += feature
                        (col, feature, -1, label)
                    } union (startCol until endCol).flatMap {col =>
                        (col+1 until endCol).map {col2 =>
                            (col, features(col), col2, features(col2))
                        }
                    }
                }
            }.countByValue()

            if (labels == null) {
                // Do this only once for the first column since labels are invariant across features.
                labels =
                        pairCounts.keys.filter(_._3 == -1).map(_._4).toArray.distinct.zipWithIndex.toMap
            }
            val numLabels = labels.size
            pairCounts.keys.groupBy(_._1).foreach { case (col, keys) =>
                val features = keys.map(_._2).toArray.distinct.zipWithIndex.toMap
                val numRows = features.size
                val featureDistinctVals = keys.filter(_._3 != -1).groupBy(_._3).mapValues(_.map(_._4).toArray.distinct.length)

                // TODO: try sparse matrices
                val featureContingencies = MMap.empty[Int, Matrix]
                val labelContingency = Matrices.zeros(numRows, numLabels)

                keys.foreach { case (_, feature1, col2, feature2) =>
                    val i = features(feature1)
                    val j = if(col2 == -1) labels(feature2) else features(feature2)
                    if(col2 == -1)
                        labelContingency(i, j) += pairCounts((col, feature1, col2, feature2))
                    else {
                        if(!featureContingencies.contains(col2))
                            // TODO: try sparse matrix
                            featureContingencies(col2) = Matrices.zeros(numRows, featureDistinctVals(col2))
                        featureContingencies(col2)(i, j) += pairCounts((col, feature1, col2, feature2))
                    }

                }
                results(col) = ???
            }
            batch += 1
        }
        results
    }
}
