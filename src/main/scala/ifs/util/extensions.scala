package ifs.util

import breeze.linalg.Matrix
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

/**
  * Object containing general utility extension methods
  */
object extensions {

    /**
      * This value class contains extension methods for [[Vector]].
      */
    implicit class RichVector(val self: Vector) extends AnyVal {

        /**
          * Maps every distinct value of the vector to its index (determined by its order of appearance).
          */
        def toIndexMap: Map[Double, Int] = distinct.zipWithIndex.toMap

        /**
          * Returns all the distinct values contained in the vector
          */
        def distinct: Seq[Double] = self.toSet.toVector

        /**
          * Converts the vector to a Set (and adds value 0.0 when appropriate).
          */
        def toSet: Set[Double] = activeValues.toSet ++ (if (self.numNonzeros < self.size) Set(0.0) else Set())

        /**
          * Returns a Seq containing all active values of the vector
          */
        def activeValues: Seq[Double] = self match {
            case v: SparseVector => v.values.toVector
            case v: DenseVector => v.toArray.toVector
        }

        /**
          * Converts the vector to a Seq
          */
        def toSeq: Seq[Double] = self.toArray.toVector

        /**
          * Returns all indices paired to their value, only if not zero
          */
        def nonZeros: Seq[(Int, Double)] = activeIndices.zip(activeValues).filter(_._2 != 0.0)

        /**
          * Returns a Seq containing all indices of active values in the vector
          */
        def activeIndices: Seq[Int] = self match {
            case v: SparseVector => v.indices.toVector
            case v: DenseVector => 0 until v.size
        }
    }

    /**
      * This value class contains extension methods for [[Matrix]] of Long integers
      */
    implicit class RichLongMatrix(val self: Matrix[Long]) extends AnyVal {

        /**
          * Returns a vector containing the sum of every column
          */
        def colSums: Seq[Long] = colRowSums._1

        /**
          * Returns a pair of vectors, the first contains the sum of every column, while the other contains
          * the sum of every row.
          */
        def colRowSums: (Seq[Long], Seq[Long]) = {
            val rowSums = Array.fill(self.rows)(0L)
            val colSums = Array.fill(self.cols)(0L)
            self foreachPair { case ((row, col), value) =>
                rowSums(row) += value
                colSums(col) += value
            }
            (colSums.toVector, rowSums.toVector)
        }

        /**
          * Returns a vector containing the sum of every row
          *
          * @return
          */
        def rowSums: Seq[Long] = colRowSums._2
    }

    /**
      * This value class contains extension methods for [[String]]
      */
    implicit class RichString(val self: String) extends AnyVal {
        /**
          * Returns a String padded to the given length with the given character.
          */
        def paddedTo(length: Int, padder: Char = ' '): String =
            self.padTo(length, padder).mkString
    }

}
