name := "spark-ifs"

version := "1.0"

scalaVersion := "2.11.12"

scalacOptions := Seq("-feature")

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-sql" % "2.3.0",
    "org.apache.spark" %% "spark-mllib" % "2.3.0",
    "org.rogach" %% "scallop" % "3.1.2"
)