package ifs.ml.stats

import breeze.linalg.Matrix
import ifs.util.extensions._
import org.apache.spark.ml.linalg.Vector

/**
  * This object can be called to compute mutual information on a contingency matrix or on two instance vectors.
  */
object MutualInformation {

    /**
      * Computes the contingency matrix of the two vectors and their mutual information.
      *
      * @param a Vector of instances.
      * @param b Another vector of instances (its length must be the same as the first).
      * @return Mutual information between a and b
      */
    def apply(a: Vector, b: Vector): Double = {

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

        apply(mat)
    }

    /**
      * Computes mutual information on the given matrix.
      *
      * @param contingency The contingency matrix of two vectors.
      * @return Mutual information of the two vectors represented by the matrix.
      */
    def apply(contingency: Matrix[Long]): Double = {

        val (colSums, rowSums) = contingency.colRowSums
        val tot = colSums.sum.toDouble

        var mi = 0.0

        contingency foreachPair { case ((row, col), value) =>
            val pxy = value / tot
            val px = rowSums(row) / tot
            val py = colSums(col) / tot

            if (pxy > 0.0) mi += pxy * Math.log(pxy / (px * py))
        }
        mi
    }
}
