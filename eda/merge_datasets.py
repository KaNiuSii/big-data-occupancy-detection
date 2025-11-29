import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType
from pyspark.sql.window import Window

DATA_DIR = "dataset"
OUTPUT_FILE = os.path.join(DATA_DIR, "occupancy_all.csv")


def create_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("MergeOccupancyDatasets")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_occupancy_file(spark: SparkSession, filename: str, set_name: str):
    path = os.path.join(DATA_DIR, filename)

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("sep", ",")
        .csv(path)
        .withColumn("set_name", F.lit(set_name))
    )

    # dopilnuj typów
    df = df.withColumn("id", F.col("id").cast("int"))
    df = df.withColumn("datetime", F.col("date").cast(TimestampType()))

    return df


def merge_and_deduplicate(spark: SparkSession):
    # 1. Wczytaj trzy zbiory
    train_df = load_occupancy_file(spark, "datatraining.txt", "train")
    test_df = load_occupancy_file(spark, "datatest.txt", "test")
    test2_df = load_occupancy_file(spark, "datatest2.txt", "test2")

    # 2. Złącz wszystko
    all_df = train_df.unionByName(test_df).unionByName(test2_df)

    print("\nPo połączeniu:")
    print(f"Liczba wierszy (łącznie): {all_df.count()}")

    # 3. Usuń duplikaty
    #    Zakładamy, że "ten sam rekord" = te same wartości cech + Occupancy.
    #    set_name i id nas nie interesują przy definicji duplikatu.
    dedup_cols = ["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]

    dedup_df = all_df.dropDuplicates(dedup_cols)

    print(f"Liczba wierszy po usunięciu duplikatów: {dedup_df.count()}")

    # 4. Nadaj nowe, globalne id (ładnie rosnące od 1)
    w = Window.orderBy("date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy")

    dedup_df = (
        dedup_df
        .withColumn("id_new", F.row_number().over(w))
        .select(
            F.col("id_new").alias("id"),
            "date",
            "Temperature",
            "Humidity",
            "Light",
            "CO2",
            "HumidityRatio",
            "Occupancy"
        )
        .orderBy("id")
    )

    # 5. Zapis do pojedynczego pliku CSV
    tmp_dir = os.path.join(DATA_DIR, "occupancy_all_tmp")

    print(f"\nZapisuję dane tymczasowo do folderu: {tmp_dir}")
    (
        dedup_df
        .coalesce(1)               # jeden plik part-*.csv
        .write
        .option("header", "true")
        .mode("overwrite")
        .csv(tmp_dir)
    )

    # Znajdź plik part-*.csv i przenieś go jako occupancy_all.csv
    part_file = None
    for fname in os.listdir(tmp_dir):
        if fname.startswith("part-") and fname.endswith(".csv"):
            part_file = fname
            break

    if part_file is None:
        raise RuntimeError("Nie znaleziono pliku part-*.csv w katalogu tymczasowym.")

    src = os.path.join(tmp_dir, part_file)
    dst = OUTPUT_FILE

    # Jeśli istnieje stary occupancy_all.csv – usuń
    if os.path.exists(dst):
        os.remove(dst)

    shutil.move(src, dst)
    print(f"Zapisano finalny plik: {dst}")

    # Sprzątanie – usuwamy resztki z katalogu tymczasowego
    for fname in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fname))
    os.rmdir(tmp_dir)

    print("\nGotowe – wszystkie rekordy (bez duplikatów) są w jednym pliku:", dst)


if __name__ == "__main__":
    spark = create_spark()
    try:
        merge_and_deduplicate(spark)
    finally:
        spark.stop()
