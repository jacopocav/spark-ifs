package ifs.util

import breeze.linalg.Matrix
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

import scala.collection.immutable.{Vector => ScalaVector}

object extensions {
    implicit class RichVector(val self: Vector) extends AnyVal {
        def toIndexMap: Map[Double, Int] = distinct.zipWithIndex.toMap

        def toSeq: Seq[Double] = self.toArray.toVector

        def toSet: Set[Double] = activeValues.toSet ++ (if(self.numNonzeros > 0) Set(0.0) else Set())

        def distinct: Seq[Double] = self.toSet.toVector

        def activeValues: Seq[Double] = self match {
            case v: SparseVector => v.values.toVector
            case v: DenseVector => v.toArray.toVector
        }

        def activeIndices: Seq[Int] = self match {
            case v: SparseVector => v.indices.toVector
            case v: DenseVector  => 0 until v.size
        }

        def nonZeros: Seq[(Int, Double)] = activeIndices.zip(activeValues).filter(_._2 != 0.0)
    }

    implicit class RichLongMatrix(val self: Matrix[Long]) extends AnyVal {
        def colRowSums: (ScalaVector[Long], ScalaVector[Long]) = {
            val rowSums = Array.fill(self.rows)(0L)
            val colSums = Array.fill(self.cols)(0L)
            self foreachPair  { case ((row, col), value) =>
                rowSums(row) += value
                colSums(col) += value
            }
            (colSums.toVector, rowSums.toVector)
        }
        def colSums: ScalaVector[Long] = colRowSums._1
        def rowSums: ScalaVector[Long] = colRowSums._2
    }
}
