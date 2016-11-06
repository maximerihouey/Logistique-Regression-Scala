# Scala Logistique Regression

The code is written in Scala 2.11 (2.11.8) and SBT 0.13 (0.13.8).

To compile the source code
<pre>
sbt clean package
</pre>

To execute the test script,
<pre>
$SPARK_HOME/bin/spark-submit --class "io.github.maximerihouey.Test" target/scala-2.11/logistic_regression_scala_2.11-1.0.jar 
</pre>
