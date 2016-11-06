package io.github.maximerihouey

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Created by maxime on 04/11/16.
  */
object Test {

  def buildGaussianDataset(size: Integer, nDims: Integer, mean: Double): Array[Array[Double]] ={
    val array = Array.ofDim[Double](size, nDims);
    for(j <- 0 to (nDims-1)){
      for(i <- 0 to (size-1)){
        array(i)(j) = mean + scala.util.Random.nextGaussian()
      }
    }
    return array;
  }

  def main(args: Array[String]) {
    ///////////////////////////////////////////////////////
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
//    val sparkSession = SparkSession.builder
//      .master("local")
//      .appName("logistic_regression_scala")
//      .getOrCreate()
    ///////////////////////////////////////////////////////

    val size = 1000

    val class_1_features = buildGaussianDataset(size, 2, -1)
    val class_1_labels = Array.fill[Double](size)(0.0)
    val class_2_features = buildGaussianDataset(size, 2, 1)
    val class_2_labels = Array.fill[Double](size)(1.0)

    val X_data = class_1_features ++ class_2_features
    val y_data = class_1_labels ++ class_2_labels


    val dataFrame = sc.parallelize(y_data zip X_data.map(row => Vectors.dense(row))).toDF("label","features")
    val splits = dataFrame.randomSplit(Array(0.8, 0.2), seed = 11L)
    val DfTrain = splits(0).cache()
    val DfTest = splits(1).cache()

//    DfTest.show()

    val fitted_model = new LogisticRegression().fit(DfTrain)
    val predictions = fitted_model.transform(DfTest)

    predictions.show()
    val nbAccurate = predictions.select("label", "prediction").map(
      row => if (row.getDouble(0) == row.getDouble(1)) 1.0 else 0.0
    ).select(sum("value")).first().get(0)

    val nbAccurateDouble : Double = nbAccurate.asInstanceOf[Double]

    println("\n")
    println("Accuracy: %f | Count: %d".format(nbAccurateDouble / predictions.count(), predictions.count()))
    println("\n")
  }
}
