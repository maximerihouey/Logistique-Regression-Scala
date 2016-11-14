package io.github.maximerihouey

/**
  * Created by maxime on 06/11/16.
  */
class LogistiqueRegression {

  var fitted: Boolean = false
  var intercept: Double = 0.0
  var coefficients: Array[Double] = null
  var featuresMultiple: Array[Array[Double]] = null
  var labels: Array[Integer] = null
  var labelsDouble: Array[Double] = null

  def fit(featuresMultiple: Array[Array[Double]], labels: Array[Integer]){
    this.featuresMultiple = featuresMultiple
    this.labels = labels
    for(i <- 0 to labels.length){
      labelsDouble(i) = labels(i).toDouble
    }
    this.coefficients = Array.ofDim[Double](featuresMultiple(0).length)

    for(k <- 0 to 10){
      val current_gradient = gradient()
      updateCoefficients(current_gradient)
      println("%d | %f".format(k, logLikelihood()))
    }
    fitted = true
  }

  def predict(featuresMultiple: Array[Array[Double]]): Array[Integer] = {
    val predictions = Array.ofDim[Integer](featuresMultiple.length)
    for(i <- 0 to (featuresMultiple.length-1)){
      predictions(i) = this.predict(featuresMultiple(i))
    }
    return predictions
  }

  def predict(features: Array[Double]): Integer = {
    val probability = posterior(features)

    if(probability > 0){
      return 1
    }else{
      return 0
    }
  }

  def logLikelihood(): Double = {
    var logLikelihoodVal: Double = 0.0
    for(i <- 0 to this.featuresMultiple.length){
      if (this.labels(i) == 1){
        logLikelihoodVal += Math.log(1 - posterior(this.featuresMultiple(i)))
      }else{
        logLikelihoodVal += Math.log(posterior(this.featuresMultiple(i)))
      }
    }

    return logLikelihoodVal
  }

  def posterior(features: Array[Double]): Double = {
    var probability = intercept
    for(i <- 0 to features.length-1){
      probability += coefficients(i) * features(i)
    }

    return probability
  }

  def gradient(): Array[Double] = {
    val gradient = Array.ofDim[Double](coefficients.length)
    for(i <- 0 to featuresMultiple.length-1){
      val posteriorVal = posterior(this.featuresMultiple(i))
      for(j <- 0 to coefficients.length-1){
        gradient(j) += (this.labelsDouble(i) - posteriorVal) * this.featuresMultiple(i)(j)
      }
    }

    return gradient
  }

  def updateCoefficients(gradient: Array[Double]) = {
    for(i <- 0 to this.coefficients.length){
      this.coefficients(i) -= gradient(i)
    }
  }
}
