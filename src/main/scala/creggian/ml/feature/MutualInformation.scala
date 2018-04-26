package creggian.ml.feature

object MutualInformation {

    def compute(matrix: Seq[Seq[Int]]): Double = {

        val tot = matrix.flatten.sum.toDouble
        val colSums = matrix.transpose.map(_.sum.toDouble)

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
    
}
