# spark-ifs: Feature Selection on Distributed Spark Datasets

This library contains algorithms for iterative feature selection on Spark. It can be used on all 3 types of distributed
datasets Spark supports (RDD, DataFrame, Dataset) and it allows the use of custom scoring functions for selection
(the default one provided is **mRMR**, Minimum Redundancy - Maximum Relevance). Included there's also a command-line
tool that allows to perform generate random integer datasets and to perform selection with mRMR on them.

This project is based on [this paper](https://arxiv.org/pdf/1709.02327.pdf) and it's a rework of [this implementation](https://github.com/creggian/spark-ifs/).

## Getting Started

The following instructions will allow you to build the library into a .jar file.

### Prerequisites

The following software must be installed on your computer:

- [Scala 2.11.12](https://www.scala-lang.org/download/2.11.12.html) (any 2.11 version should work, 2.12 is still not supported by Spark).
- [Apache Spark 2.3.0](https://spark.apache.org/downloads.html)
- [sbt 1.1.4](https://www.scala-sbt.org/download.html)

Other versions may work but have not been tested.

### Building

To build the jar, execute the following command on your terminal while on the root of this repository 
(where the `build.sbt` file is located):

```
 sbt assembly
```

The jar will be generated in the `target/scala-2.11` directory and named as `spark-ifs-assembly-X.Y.jar` 
(where X.Y is the version number, e.g. 1.0).

### Running the command-line tool

The generated jar file can be run with `spark-submit` with the following syntax:

```
spark-submit [spark arguments] spark-ifs-assembly-X.Y.jar [tool arguments]
```

[This link](https://jaceklaskowski.gitbooks.io/mastering-apache-spark/content/spark-submit.html) points to a list 
of the arguments supported by `spark-submit`.  
The following listing shows all possible arguments for the tool.

```
This program can be used to do IFS on datasets loaded from csv files (and to generate random datasets to csv).
     -h, --help   Show help message

   Subcommand: gen
   Generates a dataset with the given size.
     -a, --alt-file  <arg>   Path to the csv in alternate encoding (without the label row)
     -c, --cols  <arg>       Number of columns
     -f, --file  <arg>       Path to the csv in conventional encoding
     -l, --labels  <arg>     Path to the csv containing the label row (required for alternate encoding)
     -r, --rows  <arg>       Number of rows
     -h, --help              Show help message
   Subcommand: select
   Selects the given number of features from the provided csv datasets.
   NOTE: for this task spark-submit must be used.
     -a, --alt-file  <arg>       Path to the csv in alternate encoding (without the label row)
     -f, --file  <arg>           Path to the csv in conventional encoding
     -l, --labels  <arg>         Path to the csv containing the label row (required for alternate encoding)
     -n, --num-features  <arg>   Number of features (columns) to be selected
     -v, --verbose               Prints more information during execution
         --noverbose             Only prints the results
     -h, --help                  Show help message
```

__Note__: the `gen` subcommand does not need `spark-submit` to be run. `scala` or even `java -jar` can be used.

To include library files in your program, you can either copy them into your source directory, or you can add the
generated jar to your classpath.