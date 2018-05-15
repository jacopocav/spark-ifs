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
    final val idCol = new Param[String](this, "idCol",
        "Column that contains a unique numeric identifier for every row (i.e. feature). " +
                "To create one, spark.sql.functions.monotonicallyIncreasingID() can be used.")
    final val labelVector = new Param[Array[Double]](this, "labelVector",
        "Array containing label values associated to every column (i.e. example) in the dataset")
    setDefault(idCol -> "id")
    final val filtered = new Param[Boolean](this, "filtered",
        "If set to true, the transformed dataset will be filtered so that it will contain only selected rows " +
                "(output column will be created anyway)")

    def getNumTopRows: Int = $(numTopRows)
    setDefault(labelVector -> Array.empty)

    def getIdCol: String = $(idCol)

    def getLabelVector: Array[Double] = $(labelVector)
    setDefault(filtered -> false)

    def isFiltered: Boolean = $(filtered)
}

/**
  * Selects features using mRMR on datasets in alternate encoding (i.e. features are rows, instances are columns).
  * Columns representing instances must have been assembled into a single vector column (Spark's VectorAssembler
  * can be used for this purpose) and there must be an id column that contains unique integral identifiers for every
  * feature (The command `df.withColumn("id", monotonically_increasing_id())` can be used to generate a suitable id
  * column).
  * The output column will flag all selected rows with `true` and the rest with `false`.
  * The selector can also be optionally set to eliminate all non-selected features from the output DataFrame.
  * Use example (all the specified parameters are the default):
  * {{{
  *     import org.apache.spark.sql.functions.monotonically_increasing_id
  *     val df: DataFrame = ...
  *     val dfId = df.withColumn("id", monotonically_increasing_id()) // Adding an appropriate id column
  *     val labels: Array[Double] = ... // Labels associated to every feature (column)
  *
  *     val rs = new RowSelector()
  *                 .setNumTopRows(10)
  *                 .setIdCol("id")
  *                 .setFeaturesCol("features")
  *                 .setLabelVector(labels)
  *                 .setOutputCol("selected")
  *                 .setFiltered(false) // Non-selected rows will be kept
  *     val rsModel = fs.fit(df)
  *     val selectedDF = fsModel.transform(df)
  *
  *     val filteredDF = selectedDF.filter($"selected" === true) // Removes all non-selected rows
  * }}}
  */
final class RowSelector(override val uid: String)
        extends Estimator[RowSelectorModel] with DefaultParamsWritable with RowSelectorParams {

    /**
      * Creates an instance with random UID.
      */
    def this() = this(Identifiable.randomUID("rowSelector"))

    /**
      * Sets the number of features that should be selected (default: 10).
      */
    def setNumTopRows(value: Int): this.type = set(numTopRows -> value)

    /**
      * Sets the column containing row identifiers (default: "id").
      */
    def setIdCol(value: String): this.type = set(idCol -> value)

    /**
      * Sets the output column.
      */
    def setOutputCol(value: String): this.type = set(outputCol -> value)

    /**
      * Sets the column containing feature vectors.
      */
    def setFeaturesCol(value: String): this.type = set(featuresCol -> value)

    /**
      * Sets the labels associated to every feature (column) in the dataset (Required).
      *
      * @param value Sequence of labels ordered by column.
      */
    def setLabelVector(value: Array[Double]): this.type = set(labelVector -> value)

    /**
      * Sets whether the fitted model will filter out all non-selected rows during transformation (Default: false).
      */
    def setFiltered(value: Boolean = true): this.type = set(filtered -> value)

    /**
      * Performs selection and determines the features that must be selected.
      *
      * @param dataset The dataset whose features should be selected.
      * @return A [[RowSelectorModel]] that can be used to transform the input dataset into the output one.
      */
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

    /**
      * Validates parameters and produces the output dataset schema.
      * These checks are performed:
      * - The label vector must not be empty.
      * - The id column must exist and contain only integers or longs.
      * - The features column must exist and contain vectors.
      * - The output column must not exist.
      *
      * @param schema Schema of the Dataset the selector should be fit on.
      * @return Schema of the DataFrame returned by [[RowSelectorModel.transform()]].
      */
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

/**
  * Model returned after fitting a [[RowSelector]].
  *
  * @param selectedRows The sequence of selected feature ids (in arbitrary order).
  */
final class RowSelectorModel private[ml](override val uid: String,
                                         val selectedRows: Seq[Int])
        extends Model[RowSelectorModel] with RowSelectorParams {

    /**
      * Creates a RowSelectorModel with the given selected feature ids.
      */
    def this(selectedRows: Seq[Int]) = this(Identifiable.randomUID("rowSelectorModel"), selectedRows)

    /**
      * Sets the id column containing row identifiers.
      * (Note: by default it's set to the same value as the [[RowSelector]] that created this model).
      */
    def setIdCol(value: String): this.type = set(idCol -> value)

    /**
      * Sets the output column where the selected flags will be stored.
      * (Note: by default it's set to the same value as the [[RowSelector]] that created this model).
      */
    def setOutputCol(value: String): this.type = set(outputCol -> value)

    /**
      * Sets whether non selected features should be removed from the output DataFrame or not.
      * (Note: by default it's set to the same value as the [[RowSelector]] that created this model).
      */
    def setFiltered(value: Boolean = true): this.type = set(filtered -> value)

    /**
      * Generates the output selected flags according to [[RowSelectorModel.selectedRows]] and, if specified,
      * filters the resulting DataFrame by eliminating all non-selected rows.
      *
      * @param dataset The input Dataset (it shoud be the same Dataset the selector was fit on).
      * @return The input Dataset with an additional output column and, if requested by [[setFiltered()]],
      *         without any non-selected row.
      */
    override def transform(dataset: Dataset[_]): DataFrame = {
        transformSchema(dataset.schema, logging = true)

        val newDS = dataset.withColumn($(outputCol), dataset($(idCol)).isin(selectedRows: _*))

        if ($(filtered)) newDS.filter(newDS($(outputCol)) === true).toDF
        else newDS
    }

    /**
      * Validates parameters and returns the output dataset schema.
      */
    override def transformSchema(schema: StructType): StructType = {
        require(schema($(idCol)).dataType.isInstanceOf[NumericType], s"Column '${$(idCol)}' must be numeric")

        require(!schema.fieldNames.contains($(outputCol)), s"Column '${$(outputCol)}' already exists.")
        schema.add($(outputCol), BooleanType, nullable = false)
    }

    override def copy(extra: ParamMap): RowSelectorModel = defaultCopy(extra)
}
