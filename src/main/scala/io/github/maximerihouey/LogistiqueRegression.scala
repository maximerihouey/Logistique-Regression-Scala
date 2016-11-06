package io.github.maximerihouey

/**
  * Created by maxime on 06/11/16.
  */
class LogistiqueRegression {

  var fitted: Boolean = false
  var coefficients: Array[Double] = null

  def fit(features: Array[Array[Double]], labels: Array[Integer]){
    coefficients = Array.ofDim[Double](features(0).length);
    fitted = true
  }

  def predict(features: Array[Double]): Integer = {
    return 0;
  }

  def predict(features: Array[Array[Double]]): Array[Integer] = {
    val predictions = Array.ofDim[Integer](features.length);
    for(i <- 0 to (features.length-1)){
      predictions(i) = this.predict(features(i));
    }
    return predictions
  }
}
