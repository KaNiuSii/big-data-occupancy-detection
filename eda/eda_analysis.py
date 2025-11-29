from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def _print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def _get_numeric_cols(df: DataFrame):
    # wszystkie numeryczne, bez id i Occupancy
    return [
        c for c, t in df.dtypes
        if t in ("double", "int", "bigint") and c not in ("id", "Occupancy")
    ]


def basic_info(train_df: DataFrame, test_df: DataFrame, test2_df: DataFrame, all_df: DataFrame):
    _print_section("1. PODSTAWOWE INFORMACJE O ZBIORACH")

    for name, df in [("train", train_df), ("test", test_df), ("test2", test2_df), ("all", all_df)]:
        print(f">>> Zbiór: {name}")
        print(f"Liczba wierszy: {df.count()}")
        df.printSchema()
        print("-" * 40)

    print("\nPrzykładowe wiersze (all_df):")
    all_df.show(10, truncate=False)


def missing_values(all_df: DataFrame):
    _print_section("2. BRAKI DANYCH (NULL)")

    null_counts = all_df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in all_df.columns
    ])
    null_counts.show(truncate=False)


def numeric_statistics(all_df: DataFrame):
    _print_section("3. STATYSTYKI OPISOWE DLA CECH NUMERYCZNYCH")

    numeric_cols = _get_numeric_cols(all_df)

    print("3.1. Describe():")
    all_df.select(numeric_cols).describe().show(truncate=False)

    print("\n3.2. Percentyle (0, 25, 50, 75, 100):")
    for col in numeric_cols:
        q = all_df.approxQuantile(col, [0.0, 0.25, 0.5, 0.75, 1.0], 0.01)
        print(f"{col}: min={q[0]}, Q1={q[1]}, median={q[2]}, Q3={q[3]}, max={q[4]}")


def occupancy_distribution(train_df: DataFrame, test_df: DataFrame, test2_df: DataFrame, all_df: DataFrame):
    _print_section("4. ROZKŁAD ZMIENNEJ DOCELOWEJ OCCUPANCY")

    def show_dist(df: DataFrame, name: str):
        total = df.count()
        print(f"\n>>> Zbiór: {name} (N={total})")
        (
            df.groupBy("Occupancy")
              .agg(F.count("*").alias("count"))
              .withColumn("percentage", F.round(F.col("count") * 100.0 / total, 2))
              .orderBy("Occupancy")
              .show()
        )

    for name, df in [("train", train_df), ("test", test_df), ("test2", test2_df), ("all", all_df)]:
        show_dist(df, name)


def means_by_occupancy(all_df: DataFrame):
    _print_section("5. ŚREDNIE WARTOŚCI CECH DLA OCCUPANCY=0 / 1")

    numeric_cols = _get_numeric_cols(all_df)

    means_by_occ = (
        all_df.groupBy("Occupancy")
              .agg(*[
                  F.round(F.avg(c), 3).alias(f"avg_{c}")
                  for c in numeric_cols
              ])
              .orderBy("Occupancy")
    )

    means_by_occ.show(truncate=False)


def time_based_analysis(all_df: DataFrame):
    _print_section("6. ANALIZA CZASOWA (GODZINY / DNI TYGODNIA)")

    # upewniamy się, że datetime nie jest cały NULL
    non_null_datetimes = all_df.filter(F.col("datetime").isNotNull()).limit(1).count()
    if non_null_datetimes == 0:
        print("Uwaga: kolumna datetime jest cała NULL – brak analizy czasowej.")
        return

    df = (
        all_df
        .withColumn("hour", F.hour("datetime"))
        .withColumn("day_of_week_spark", F.dayofweek("datetime"))
        .withColumn(
            "day_of_week",
            ((F.col("day_of_week_spark") + 5) % 7) + 1
        )
    )

    print("6.1. Średnie Occupancy wg godziny:")
    (
        df.groupBy("hour")
          .agg(
              F.count("*").alias("n"),
              F.avg(F.col("Occupancy").cast("double")).alias("occupancy_rate")
          )
          .orderBy("hour")
          .show(24, truncate=False)
    )

    print("\n6.2. Średnie Occupancy wg dnia tygodnia (1=pon,7=niedz):")
    (
        df.groupBy("day_of_week")
          .agg(
              F.count("*").alias("n"),
              F.avg(F.col("Occupancy").cast("double")).alias("occupancy_rate")
          )
          .orderBy("day_of_week")
          .show(7, truncate=False)
    )


def correlations(all_df: DataFrame):
    _print_section("7. KORELACJE ZMIENNYCH NUMERYCZNYCH Z OCCUPANCY")

    numeric_cols = _get_numeric_cols(all_df)
    df = all_df.withColumn("Occupancy_double", F.col("Occupancy").cast("double"))

    for col in numeric_cols:
        corr_val = df.stat.corr(col, "Occupancy_double")
        print(f"corr(Occupancy, {col}) = {corr_val}")


def simple_rules(all_df: DataFrame, light_threshold: float = 300.0, co2_threshold: float = 1000.0):
    _print_section("8. PROSTE REGUŁY NA PODSTAWIE LIGHT I CO2")

    print(f"8.1. Occupancy rate dla Light > {light_threshold}:")
    (
        all_df.filter(F.col("Light") > light_threshold)
              .agg(F.avg(F.col("Occupancy").cast("double")).alias("occupancy_rate"))
              .show()
    )

    print(f"\n8.2. Occupancy rate dla CO2 > {co2_threshold}:")
    (
        all_df.filter(F.col("CO2") > co2_threshold)
              .agg(F.avg(F.col("Occupancy").cast("double")).alias("occupancy_rate"))
              .show()
    )

    print(f"\n8.3. Occupancy rate dla Light <= {light_threshold} i CO2 <= {co2_threshold}:")
    (
        all_df.filter((F.col("Light") <= light_threshold) & (F.col("CO2") <= co2_threshold))
              .agg(F.avg(F.col("Occupancy").cast("double")).alias("occupancy_rate"))
              .show()
    )


def run_full_eda(train_df: DataFrame, test_df: DataFrame, test2_df: DataFrame):
    """
    Główna funkcja – odpalasz ją z main.py.
    Robi pełną eksploracyjną analizę danych i wypisuje wszystko na konsolę.
    """
    all_df = train_df.unionByName(test_df).unionByName(test2_df)

    basic_info(train_df, test_df, test2_df, all_df)
    missing_values(all_df)
    numeric_statistics(all_df)
    occupancy_distribution(train_df, test_df, test2_df, all_df)
    means_by_occupancy(all_df)
    time_based_analysis(all_df)
    correlations(all_df)
    simple_rules(all_df)
