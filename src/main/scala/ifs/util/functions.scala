package ifs.util

import java.io.FileWriter

import scala.util.Random

/**
  * Object containing general utility functions
  */
object functions {
    /**
      * Executes and times the given code
      * @param block The block of code to execute
      * @return A pair containing the result and the total time
      */
    def doAndTime[T](block: => T): (T, Long) = {
        val start = System.nanoTime()
        val ret = block
        val end = System.nanoTime()
        (ret, (end - start) / 1000000)
    }

    /**
      * Prints a line if verbose is set to true. By using an implicit parameter, this function can be
      * invoked just like a normal [[println]].
      * @param x Stuff to print in a line
      * @param verbose If false, this function does nothing
      */
    def verbosePrintln(x: Any)(implicit verbose: Boolean): Unit = {
        if(verbose) println(x)
    }

    /**
      * The equivalent of [[verbosePrintln]] for [[print]].
      */
    def verbosePrint(x: Any)(implicit verbose: Boolean): Unit = {
        if(verbose) print(x)
    }


    /**
      * Generates a random matrix of single-digit integers and saves it to file in csv format
      * using either the conventional encoding or the alternate one (or both).
      * @param rows Number of rows of the desired matrix
      * @param cols Number of columns of the desired matrix
      * @param convFile If it's not None, then the conventional encoded matrix is saved to this path.
      * @param altFiles If it's not None, then the alternate encoded matrix is saved to the first path,
      *                 and the label row is saved to the second one.
      */
    def randomMatricesToCsv(rows: Int, cols: Int, convFile: Option[String], altFiles: Option[(String, String)]): Unit = {
        val data = Vector.fill(rows, cols)(Random.nextInt(10))

        val csv = data.map(_.mkString(",")).mkString("\n")

        convFile foreach { file =>
            val convWriter = new FileWriter(file)
            convWriter.write(csv)
            convWriter.close()
        }

        altFiles foreach { case (matrixFile, labelFile) =>
            val altCsv = data.transpose.map(_.mkString(","))

            val altWriter = new FileWriter(matrixFile)
            altWriter.write(altCsv.drop(1).mkString("\n"))
            altWriter.close()

            val labWriter = new FileWriter(labelFile)
            labWriter.write(altCsv(0))
            labWriter.close()
        }
    }
}
