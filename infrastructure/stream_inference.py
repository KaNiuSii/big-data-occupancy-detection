from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

from pathlib import Path

PROJECT_ROOT = Path("/opt/app")
MODEL_PATH = str(PROJECT_ROOT / "ml" / "models" / "occupancy_lr")

KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
REQUEST_TOPIC = "occupancy_requests"
RESPONSE_TOPIC = "occupancy_responses"


def create_spark():
    spark = (
        SparkSession.builder
        .appName("OccupancyStreamingInference")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def build_request_schema():
    payload_schema = StructType([
        StructField("Temperature", DoubleType(), nullable=True),   # zmieniam na True
        StructField("Humidity", DoubleType(), nullable=True),
        StructField("Light", DoubleType(), nullable=True),
        StructField("CO2", DoubleType(), nullable=True),
        StructField("HumidityRatio", DoubleType(), nullable=True),
    ])

    request_schema = StructType([
        StructField("request_id", StringType(), nullable=False),
        StructField("timestamp", StringType(), nullable=True),
        StructField("payload", payload_schema, nullable=True),
    ])

    return request_schema


def main():
    spark = create_spark()
    schema = build_request_schema()

    print("=== Ładuję model z MLlib ===")
    model = PipelineModel.load(MODEL_PATH)

    print("=== Konfiguruję strumień z Kafki (requests) ===")
    raw_stream = (
        spark.readStream
             .format("kafka")
             .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
             .option("subscribe", REQUEST_TOPIC)
             .option("startingOffsets", "latest")
             .load()
    )

    json_stream = (
        raw_stream
        .select(F.col("value").cast("string").alias("value_str"))
        .select(F.from_json("value_str", schema).alias("data"))
    )

    parsed = (
        json_stream
        .select(
            F.col("data.request_id").alias("request_id"),
            F.col("data.timestamp").alias("timestamp"),
            F.col("data.payload.Temperature").alias("Temperature"),
            F.col("data.payload.Humidity").alias("Humidity"),
            F.col("data.payload.Light").alias("Light"),
            F.col("data.payload.CO2").alias("CO2"),
            F.col("data.payload.HumidityRatio").alias("HumidityRatio"),
        )
    )

    # === WALIDACJA WEJŚCIA ===
    feature_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

    # is_valid = wszystkie cechy nie są null
    cond = F.lit(True)
    for c in feature_cols:
        cond = cond & F.col(c).isNotNull()

    parsed_flagged = parsed.withColumn("is_valid", cond)

    valid = parsed_flagged.filter("is_valid")
    invalid = parsed_flagged.filter("NOT is_valid")

    # === DLA POPRAWNYCH DANYCH – normalne inferowanie modelem ===
    preds_valid = model.transform(valid.drop("is_valid"))

    preds_valid = preds_valid.withColumn(
        "probability_arr",
        vector_to_array("probability")
    )

    result_valid = preds_valid.select(
        "request_id",
        "timestamp",
        "Temperature",
        "Humidity",
        "Light",
        "CO2",
        "HumidityRatio",
        F.col("prediction").cast("int").alias("prediction"),
        (F.col("probability_arr")[1]).alias("probability"),
    )

    # === DLA ZŁYCH DANYCH – SENTINEL (-1) ZAMIAST WYJAZDU STRUMIENIA ===
    # Możesz tu też dodać np. osobne pole "error" albo "reason"
    invalid_result = (
        invalid
        .select(
            "request_id",
            "timestamp",
            "Temperature",
            "Humidity",
            "Light",
            "CO2",
            "HumidityRatio",
        )
        .withColumn("prediction", F.lit(-1).cast("int"))
        .withColumn("probability", F.lit(-1.0))
    )

    # Łączymy obie ścieżki w jeden strumień wynikowy
    result = result_valid.unionByName(invalid_result)

    # === JSON odpowiedzi ===
    response_json = result.select(
        F.col("request_id"),
        F.to_json(
            F.struct(
                F.col("request_id"),
                F.col("timestamp"),
                F.struct(
                    F.col("Temperature"),
                    F.col("Humidity"),
                    F.col("Light"),
                    F.col("CO2"),
                    F.col("HumidityRatio"),
                ).alias("features"),
                F.col("prediction"),
                F.col("probability"),
            )
        ).alias("value")
    )

    kafka_out = (
        response_json
        .selectExpr(
            "CAST(request_id AS STRING) AS key",
            "CAST(value AS STRING) AS value"
        )
    )

    print("=== Startuję strumień do Kafki (responses) ===")
    query = (
        kafka_out
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", RESPONSE_TOPIC)
        .option("checkpointLocation", "/tmp/checkpoints/occupancy_inference")
        .outputMode("append")
        .start()
    )

    # Debug: pokazujemy też cały wynik (łącznie z invalid)
    debug_query = (
        result
        .writeStream
        .format("console")
        .option("truncate", False)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()
    debug_query.awaitTermination()


if __name__ == "__main__":
    main()
