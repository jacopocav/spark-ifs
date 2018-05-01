package creggian.ml.feature

import breeze.linalg.Matrix


object MutualInformation {

    def compute(matrix: Seq[Seq[Long]]): Double = {

        val colSums = matrix.transpose.map(_.sum.toDouble)
        val tot = colSums.sum

        var mi = 0.0

        for (row <- matrix) {
            for ((el, i) <- row.zipWithIndex) {

                val pxy = el.toDouble / tot
                val px  = row.sum.toDouble / tot
                val py  = colSums(i) / tot

                if (pxy > 0.0) {
                    mi += pxy * Math.log(pxy / (px * py))
                }
            }
        }
        mi
    }

    def compute(a: Seq[Double], b: Seq[Double]): Double = {
        require(a.length == b.length, s"Vectors a and b must have the same length: ${a.length} != ${b.length}")

        val ab = b.zip(a)
        val abLevels = ab.distinct

        var mi = 0.0
        for ((bl, al) <- abLevels) {
            val pxy = ab.count(_ == (bl, al)).toDouble / a.length
            val px  = ab.count(_._1 == bl).toDouble / a.length
            val py  = ab.count(_._2 == al).toDouble / a.length

            if (pxy > 0.0) mi += pxy * Math.log(pxy / (px * py))
        }
        mi
    }

    // Extension method(s) for Matrix class
    private implicit class RichLongMatrix(val self: Matrix[Long]) extends AnyVal {
        def colRowSums: (Vector[Long], Vector[Long]) = {
            val rowSums = Array.fill(self.rows)(0L)
            val colSums = Array.fill(self.cols)(0L)
            self foreachPair  { case ((row, col), value) =>
                rowSums(row) += value
                colSums(col) += value
            }
            (colSums.toVector, rowSums.toVector)
        }
        def colSums: Vector[Long] = colRowSums._1
        def rowSums: Vector[Long] = colRowSums._2
    }

    def compute(matrix: Matrix[Long]): Double = {

        val (colSums, rowSums) = matrix.colRowSums
        val tot = colSums.sum.toDouble

        var mi = 0.0

        matrix foreachPair  {case ((row, col), value) =>
            val pxy = value / tot
            val px = rowSums(row) / tot
            val py = colSums(col) / tot

            if(pxy > 0.0) mi += pxy * Math.log(pxy / (px * py))
        }
        mi
    }
    
}
