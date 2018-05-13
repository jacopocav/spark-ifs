package ifs.ml.feature

import org.apache.spark.ml.feature.{LabeledPoint, VectorSlicer}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DataTypes, NumericType, StructType}
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

class FeatureSelector(override val uid: String) extends Estimator[FeatureSelectorModel] with FeatureSelectorParams with DefaultParamsWritable {

    def this() = this(Identifiable.randomUID("featureSelector"))

    def setNumTopFeatures(value: Int): this.type = set(numTopFeatures -> value)

    def setLabelCol(value: String): this.type = set(labelCol -> value)

    def setFeaturesCol(value: String): this.type = set(featuresCol -> value)

    def setOutputCol(value: String): this.type = set(outputCol -> value)

    override def fit(dataset: Dataset[_]): FeatureSelectorModel = {
        transformSchema(dataset.schema, logging = true)

        val input: RDD[LabeledPoint] = dataset.select(dataset($(labelCol)).cast(DataTypes.DoubleType), dataset($(featuresCol))).rdd.map {
            case Row(label: Double, features: Vector) => LabeledPoint(label, features)
        }
        val selectedFeatures = IterativeFeatureSelection.selectColumns(input, $(numTopFeatures)).map(_._1)
        new FeatureSelectorModel(selectedFeatures).setParent(this)
                .setFeaturesCol($(featuresCol))
                .setOutputCol($(outputCol))
    }

    override def transformSchema(schema: StructType): StructType = {
        // Column type validation
        require(schema($(labelCol)).dataType.isInstanceOf[NumericType], s"Values in column '${$(labelCol)}' must be numeric.")
        require(schema($(featuresCol)).dataType == VectorType, s"Column '${$(featuresCol)}' must contain vectors.")
        require(!schema.fieldNames.contains($(outputCol)), s"Column '${$(outputCol)}' already exists.")

        schema.add($(outputCol), VectorType, nullable = false)
    }

    override def copy(extra: ParamMap): Estimator[FeatureSelectorModel] = defaultCopy(extra)
}

class FeatureSelectorModel private[ml](override val uid: String, val selectedFeatures: Seq[Int])
        extends Model[FeatureSelectorModel] with FeatureSelectorParams {

    private val slicer = new VectorSlicer().setIndices(selectedFeatures.sorted.toArray)

    def this(selectedFeatures: Seq[Int]) = this(Identifiable.randomUID("featureSelectorModel"), selectedFeatures)

    def setFeaturesCol(value: String): this.type = {
        slicer.setInputCol(value)
        set(featuresCol -> value)
    }

    def setOutputCol(value: String): this.type = {
        slicer.setOutputCol(value)
        set(outputCol -> value)
    }

    override def transform(dataset: Dataset[_]): DataFrame = slicer.transform(dataset)

    override def transformSchema(schema: StructType): StructType = slicer.transformSchema(schema)

    override def copy(extra: ParamMap): FeatureSelectorModel = defaultCopy(extra)
}