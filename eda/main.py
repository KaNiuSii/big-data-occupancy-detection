from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType

# --------------------------------------------------
# 1. Inicjalizacja SparkSession
# --------------------------------------------------
spark = (
    SparkSession.builder
    .appName("OccupancyDetectionEDA")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

DATA_DIR = "dataset"
