package ifs.ml.feature.stats

import breeze.linalg.Matrix
import ifs.util.implicits._


object MutualInformation {

    def compute(contingency: Matrix[Long]): Double = {

        val (colSums, rowSums) = contingency.colRowSums
        val tot = colSums.sum.toDouble

        var mi = 0.0

        contingency foreachPair  {case ((row, col), value) =>
            val pxy = value / tot
            val px = rowSums(row) / tot
            val py = colSums(col) / tot

            if(pxy > 0.0) mi += pxy * Math.log(pxy / (px * py))
        }
        mi
    }
    
}
