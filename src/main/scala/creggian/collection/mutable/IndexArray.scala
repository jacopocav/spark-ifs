package creggian.collection.mutable

import scala.collection.mutable

trait IndexArray extends Serializable {

    def length: Int

    def add(index: Int): this.type

    final def +=(index: Int): this.type = add(index)

    def remove(index: Int): this.type

    final def -=(index: Int): this.type = remove(index)

    def contains(index: Int): Boolean

    final def apply(index: Int): Boolean = contains(index)

    def indices: Seq[Int]

    def fill(): this.type
}

class SparseIndexArray(override val length: Int) extends IndexArray {
    private val indexSet: mutable.Set[Int] = mutable.SortedSet.empty

    def add(index: Int): SparseIndexArray.this.type = {
        if (index < length) indexSet += index
        this
    }

    def remove(index: Int): SparseIndexArray.this.type = {
        if (index < length) indexSet -= index
        this
    }

    def contains(index: Int): Boolean = indexSet.contains(index)

    def indices: Seq[Int] = indexSet.toSeq

    def fill(): SparseIndexArray.this.type = {
        indexSet ++= (0 until length)
        this
    }
}

class DenseIndexArray(override val length: Int) extends IndexArray {

    private val indexSet: mutable.Set[Int] = mutable.SortedSet(0 until length: _*)

    def add(index: Int): this.type = {
        if (index < length) indexSet -= index
        this
    }

    def remove(index: Int): this.type = {
        if (index < length) indexSet += index
        this
    }

    def contains(index: Int): Boolean = index < length && !indexSet.contains(index)

    def indices: Seq[Int] = (0 until length).filter(!indexSet.contains(_))

    def fill(): this.type = {
        indexSet.clear()
        this
    }
}

object IndexArray {
    def dense(length: Int) = new DenseIndexArray(length)

    def sparse(length: Int) = new SparseIndexArray(length)
}