package ifs.ml.feature

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasOutputCol}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

private[feature] trait RowSelectorParams extends DefaultParamsWritable with HasFeaturesCol with HasOutputCol {

    final val numTopRows = new Param[Int](this, "numTopRows",
        "Number of rows to select.",
        ParamValidators.gt(0))
    setDefault(numTopRows -> 10)

    def getNumTopRows: Int = $(numTopRows)

    final val idCol = new Param[String](this, "idCol",
        "Column that contains a unique numeric identifier for every row (i.e. feature). " +
                "To create one, spark.sql.functions.monotonicallyIncreasingID() can be used.")
    setDefault(idCol -> "id")

    def getIdCol: String = $(idCol)

    final val labelVector = new Param[Array[Double]](this, "labelVector",
        "Array containing label values associated to every column (i.e. example) in the dataset")
    setDefault(labelVector -> Array.empty)

    def getLabelVector: Array[Double] = $(labelVector)

    final val filtered = new Param[Boolean](this, "filtered",
        "If set to true, the transformed dataset will be filtered so that it will contain only selected rows " +
            "(output column will be created anyway)")
    setDefault(filtered -> false)

    def isFiltered: Boolean = $(filtered)
}

final class RowSelector(override val uid: String)
        extends Estimator[RowSelectorModel] with DefaultParamsWritable with RowSelectorParams {

    def this() = this(Identifiable.randomUID("rowSelector"))

    def setNumTopRows(value: Int): this.type = set(numTopRows -> value)

    def setIdCol(value: String): this.type = set(idCol -> value)

    def setOutputCol(value: String): this.type = set(outputCol -> value)

    def setFeaturesCol(value: String): this.type = set(featuresCol -> value)

    def setLabelVector(value: Array[Double]): this.type = set(labelVector -> value)

    def setFiltered(value: Boolean = true): this.type = set(filtered -> value)

    override def fit(dataset: Dataset[_]): RowSelectorModel = {
        transformSchema(dataset.schema, logging = true)

        val input: RDD[LabeledPoint] = dataset.select(dataset($(idCol)).cast(DoubleType), dataset($(featuresCol)))
                .rdd.map {
            case Row(id: Double, features: Vector) => LabeledPoint(id, features)
        }

        val labVec = Vectors.dense($(labelVector))

        val selected = IterativeFeatureSelection.selectRows(input, $(numTopRows), labVec).map(_._1.toInt)

        new RowSelectorModel(selected).setParent(this)
                .setIdCol($(idCol))
                .setFiltered($(filtered))
                .setOutputCol($(outputCol))
    }

    override def transformSchema(schema: StructType): StructType = {
        require(schema($(featuresCol)).dataType == VectorType, s"Column '${$(featuresCol)}' must contain vectors.")
        require(schema($(idCol)).dataType == LongType
                | schema($(idCol)).dataType == IntegerType, s"Column '${$(idCol)}' must be integral.")

        require($(labelVector).nonEmpty, s"labelVector is empty.")

        require(!schema.fieldNames.contains($(outputCol)), s"Column '${$(outputCol)}' already exists.")
        schema.add($(outputCol), BooleanType, nullable = false)
    }

    override def copy(extra: ParamMap): Estimator[RowSelectorModel] = defaultCopy(extra)
}


final class RowSelectorModel private[ml](override val uid: String,
                                   val selectedRows: Seq[Int])
        extends Model[RowSelectorModel] with RowSelectorParams {

    def this(selectedRows: Seq[Int]) = this(Identifiable.randomUID("rowSelectorModel"), selectedRows)

    def setIdCol(value: String): this.type = set(idCol -> value)

    def setOutputCol(value: String): this.type = set(outputCol -> value)

    def setFiltered(value: Boolean = true): this.type = set(filtered -> value)


    override def transform(dataset: Dataset[_]): DataFrame = {
        transformSchema(dataset.schema, logging = true)

        val newDS = dataset.withColumn($(outputCol), dataset($(idCol)).isin(selectedRows:_*))

        if($(filtered)) newDS.filter(newDS($(outputCol)) === true).toDF
        else newDS
    }

    override def transformSchema(schema: StructType): StructType = {
        require(schema($(idCol)).dataType.isInstanceOf[NumericType], s"Column '${$(idCol)}' must be numeric")

        require(!schema.fieldNames.contains($(outputCol)), s"Column '${$(outputCol)}' already exists.")
        schema.add($(outputCol), BooleanType, nullable = false)
    }

    override def copy(extra: ParamMap): RowSelectorModel = defaultCopy(extra)
}
