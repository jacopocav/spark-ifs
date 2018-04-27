/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package creggian.ml.feature.algorithm

import creggian.ml.feature.MutualInformation
import org.apache.spark.ml.linalg.Matrix

object InstanceMRMR extends InstanceWiseScore {

    def getResult(matWithClass: Seq[Seq[Int]], matWithFeatures: Seq[Seq[Seq[Int]]], selectedVariablesIdx: Seq[Int], variableLevels: Seq[Double], classLevels: Seq[Double], i: Int, nfs: Int): Double = {
        mrmrMutualInformation(matWithClass, matWithFeatures, selectedVariablesIdx)
    }

    private def mrmrMutualInformation(matWithClass: Seq[Seq[Int]], matWithFeatures: Seq[Seq[Seq[Int]]], selectedVariablesIdx: Seq[Int]): Double = {

        val mrmrClass = MutualInformation.compute(matWithClass)

        val mrmrFeatures = matWithFeatures.foldLeft(0.0)((a, f) => a + MutualInformation.compute(f))

        val coefficient =
            if (selectedVariablesIdx.length > 1)
                1.0 / selectedVariablesIdx.length
            else
                1.0

        mrmrClass - (coefficient * mrmrFeatures)
    }

    override def selectTop(i: Int, nfs: Int): Int = 1

    override def getResult(labelContingency: Matrix, featuresContingencies: Traversable[Matrix]): Double = {
        val labelScore = MutualInformation.compute(labelContingency.toArray.map(_.toInt).toSeq.grouped(labelContingency.numCols).toSeq)

        val featuresScore = featuresContingencies.foldLeft(0.0) { (acc, mat) =>
            acc + MutualInformation.compute(mat.toArray.map(_.toInt).toSeq.grouped(mat.numCols).toSeq)
        }

        val coefficient
    }
}
