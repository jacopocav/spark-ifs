package creggian.ml.feature

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWritable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

private[feature] trait FeatureSelectorParams extends Params
        with HasFeaturesCol with HasOutputCol with HasLabelCol {

}

class FeatureSelector(override val uid: String) extends Estimator[FeatureSelectorModel] with FeatureSelectorParams with DefaultParamsWritable {
    override def fit(dataset: Dataset[_]): FeatureSelectorModel = ???

    override def transformSchema(schema: StructType): StructType = ???

    override def copy(extra: ParamMap): Estimator[FeatureSelectorModel] = defaultCopy(extra)
}

class FeatureSelectorModel private[ml] (override val uid: String)
        extends Model[FeatureSelectorModel] with FeatureSelectorParams {

    override def transform(dataset: Dataset[_]): DataFrame = ???

    override def transformSchema(schema: StructType): StructType = ???

    override def copy(extra: ParamMap): FeatureSelectorModel = defaultCopy(extra)
}