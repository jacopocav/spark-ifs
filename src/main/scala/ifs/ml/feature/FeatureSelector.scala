package ifs.ml.feature

import org.apache.spark.ml.feature.{LabeledPoint, VectorSlicer}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

private[feature] trait FeatureSelectorParams extends Params
        with HasFeaturesCol with HasOutputCol with HasLabelCol {

    final val numTopFeatures = new Param[Int](this, "numTopFeatures",
        "Number of features that will be selected" +
                "if total features < numSelectedFeatures, then all features will be selected",
        ParamValidators.gt(0))
    setDefault(numTopFeatures -> 10)

    def getNumSelectedFeatures: Int = $(numTopFeatures)
}

/**
  * Selects features using mRMR on datasets in conventional encoding (i.e. features are columns, instances are rows).
  * Columns representing features must have been assembled into a single vector column (Spark's VectorAssembler
  * can be used for this purpose) and the label column associated to instances must be numeric
  * (StringIndexer can be used to make string labels numeric, and IndexToString to convert them back to strings).
  * The output vector column will contain only the specified number of selected features.
  * Use example (all the specified parameters are the default):
  * {{{
  *     val df: DataFrame = ...
  *     val fs = new FeatureSelector()
  *                 .setNumTopFeatures(10)
  *                 .setLabelCol("label")
  *                 .setFeaturesCol("features")
  *                 .setOutputCol("selected")
  *     val fsModel = fs.fit(df)
  *     val selectedDF = fsModel.transform(df)
  * }}}
  */
class FeatureSelector(override val uid: String) extends Estimator[FeatureSelectorModel]
        with FeatureSelectorParams with DefaultParamsWritable {

    /**
      * Constructs a FeatureSelector with a random UID
      */
    def this() = this(Identifiable.randomUID("featureSelector"))

    /**
      * Sets the number of features that should be selected (default: 10)
      */
    def setNumTopFeatures(value: Int): this.type = set(numTopFeatures -> value)

    /**
      * Sets the the column containing all the numeric labels (default: "label")
      */
    def setLabelCol(value: String): this.type = set(labelCol -> value)

    /**
      * Sets the column containing all feature vectors (default: "features")
      */
    def setFeaturesCol(value: String): this.type = set(featuresCol -> value)

    /**
      * Sets the column where the resulting feature vectors should be stored (default: "selected").
      * Note: the input Dataset must not already contain a column with the same name.
      */
    def setOutputCol(value: String): this.type = set(outputCol -> value)

    /**
      * Performs selection and determines the features that must be selected.
      *
      * @param dataset The dataset where selection should take place.
      * @return A [[FeatureSelectorModel]] that can be used to transform the input dataset into the output one.
      */
    override def fit(dataset: Dataset[_]): FeatureSelectorModel = {
        transformSchema(dataset.schema, logging = true)

        val input: RDD[LabeledPoint] = dataset.select(dataset($(labelCol)).cast(DoubleType), dataset($(featuresCol))).rdd.map {
            case Row(label: Double, features: Vector) => LabeledPoint(label, features)
        }
        val selectedFeatures = IterativeFeatureSelection.selectColumns(input, $(numTopFeatures)).map(_._1)
        new FeatureSelectorModel(selectedFeatures).setParent(this)
                .setFeaturesCol($(featuresCol))
                .setOutputCol($(outputCol))
    }

    /**
      * Validates parameters and produces the output dataset schema.
      * These checks are performed:
      * - The label column must exist and contain either integers or longs.
      * - The features column must exist and contain vectors.
      * - The output column must not exist.
      *
      * @param schema Schema of the dataset the selector should be fit on.
      * @return Schema of the DataFrame returned by [[FeatureSelectorModel.transform()]].
      */
    override def transformSchema(schema: StructType): StructType = {
        // Column type validation
        require(schema($(labelCol)).dataType == IntegerType
                | schema($(labelCol)).dataType == LongType, s"Values in column '${$(labelCol)}' must be integral.")
        require(schema($(featuresCol)).dataType == VectorType, s"Column '${$(featuresCol)}' must contain vectors.")
        require(!schema.fieldNames.contains($(outputCol)), s"Column '${$(outputCol)}' already exists.")

        schema.add($(outputCol), VectorType, nullable = false)
    }

    override def copy(extra: ParamMap): Estimator[FeatureSelectorModel] = defaultCopy(extra)
}

/**
  * Model returned after fitting a [[FeatureSelector]].
  *
  * @param selectedFeatures The sequence of selected feature indices (in arbitrary order).
  */
class FeatureSelectorModel private[ml](override val uid: String, val selectedFeatures: Seq[Int])
        extends Model[FeatureSelectorModel] with FeatureSelectorParams {

    private val slicer = new VectorSlicer().setIndices(selectedFeatures.sorted.toArray)

    /**
      * Constructs a FeatureSelectorModel with the given selected feature indices.
      */
    def this(selectedFeatures: Seq[Int]) = this(Identifiable.randomUID("featureSelectorModel"), selectedFeatures)

    /**
      * Sets the feature column that should be transformed
      * (Note: by default it's set to the same value as the [[FeatureSelector]] that created this model).
      */
    def setFeaturesCol(value: String): this.type = {
        slicer.setInputCol(value)
        set(featuresCol -> value)
    }

    /**
      * Sets the output column where the transformation should be stored.
      * (Note: by default it's set to the same value as the [[FeatureSelector]] that created this model).
      */
    def setOutputCol(value: String): this.type = {
        slicer.setOutputCol(value)
        set(outputCol -> value)
    }

    /**
      * Generates the output feature vectors in the given column by slicing the original vectors according to
      * [[FeatureSelectorModel.selectedFeatures]].
      *
      * @param dataset The input dataset (it should contain the same features that have been fit by [[FeatureSelector]])
      * @return The input dataset with an additional output column.
      */
    override def transform(dataset: Dataset[_]): DataFrame = slicer.transform(dataset)

    /**
      * Validates parameters and returns the output dataset schema.
      */
    override def transformSchema(schema: StructType): StructType = slicer.transformSchema(schema)

    override def copy(extra: ParamMap): FeatureSelectorModel = defaultCopy(extra)
}