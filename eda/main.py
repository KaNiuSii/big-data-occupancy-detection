from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

from eda_analysis import run_full_eda

spark = (
    SparkSession.builder
    .appName("OccupancyDetectionEDA")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

DATA_DIR = "dataset"


def load_occupancy_file(filename: str, set_name: str):
    path = f"{DATA_DIR}/{filename}"

    df = (
        spark.read
        .option("header", "true")      # teraz nagłówek jest poprawny
        .option("inferSchema", "true") # Spark sam rozpozna typy
        .option("sep", ",")
        .csv(path)
        .withColumn("set_name", F.lit(set_name))
    )

    # Upewniamy się, że id jest intem (jakby Spark dał long/double)
    df = df.withColumn("id", F.col("id").cast("int"))

    # data -> Timestamp, bez kombinacji z unix_timestamp
    df = df.withColumn("datetime", F.col("date").cast(TimestampType()))

    return df

if __name__ == "__main__":
    train_df = load_occupancy_file("datatraining.txt", "train")
    test_df = load_occupancy_file("datatest.txt", "test")
    test2_df = load_occupancy_file("datatest2.txt", "test2")

    train_df.printSchema()
    train_df.show(2)

    test_df.printSchema()
    test_df.show(2)

    test2_df.printSchema()
    test2_df.show(2)

    run_full_eda(train_df, test_df, test2_df)

    spark.stop()


