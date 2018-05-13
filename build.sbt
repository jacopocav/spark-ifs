name := "spark-ifs"

version := "1.0"

scalaVersion := "2.11.12"

scalacOptions := Seq("-feature")

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-sql" % "2.3.0" % "provided",
    "org.apache.spark" %% "spark-mllib" % "2.3.0" % "provided",
    "org.rogach" %% "scallop" % "3.1.2"
)

assemblyMergeStrategy in assembly := {
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x =>
        val oldStrategy = (assemblyMergeStrategy in assembly).value
        oldStrategy(x)
}