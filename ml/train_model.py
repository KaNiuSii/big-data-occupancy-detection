import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

DATA_PATH = "dataset/occupancy_all.csv"
MODEL_PATH = "models/occupancy_lr"
REPORT_DIR = "reports"


# =====================================================================================
# Pomocnicze
# =====================================================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_and_save_confusion_matrix(cm: np.ndarray, labels, dataset_name: str):
    """
    cm: 2x2 macierz numpy
    labels: np. ['0', '1']
    """
    ensure_dir(REPORT_DIR)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion matrix ({dataset_name})")
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Rzeczywista")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # wartości w komórkach
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    out_path = os.path.join(REPORT_DIR, f"confusion_matrix_{dataset_name}.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Zapisano wykres confusion matrix: {out_path}")

    # zapis do CSV
    df_cm = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels]
    )
    csv_path = os.path.join(REPORT_DIR, f"confusion_matrix_{dataset_name}.csv")
    df_cm.to_csv(csv_path)
    print(f"Zapisano confusion matrix jako CSV: {csv_path}")


def save_metrics(metrics: dict, dataset_name: str):
    ensure_dir(REPORT_DIR)
    df = pd.DataFrame(
        [metrics],
        columns=metrics.keys()
    )
    out_path = os.path.join(REPORT_DIR, f"metrics_{dataset_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Zapisano metryki do: {out_path}")


def plot_and_save_curve(pdf: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str):
    ensure_dir(REPORT_DIR)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pdf[x_col], pdf[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True)

    out_path = os.path.join(REPORT_DIR, filename)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Zapisano wykres: {out_path}")

    # zapis CSV
    csv_path = os.path.join(REPORT_DIR, filename.replace(".png", ".csv"))
    pdf.to_csv(csv_path, index=False)
    print(f"Zapisano dane krzywej do: {csv_path}")


# =====================================================================================
# Spark + model
# =====================================================================================

def create_spark():
    spark = (
        SparkSession.builder
        .appName("OccupancyBatchTrain")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark: SparkSession):
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(DATA_PATH)
    )

    print("\n=== Wczytane dane ===")
    df.printSchema()
    df.show(10, truncate=False)

    df = df.withColumn("Occupancy", F.col("Occupancy").cast("int"))
    df = df.withColumnRenamed("Occupancy", "label")

    feature_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

    print("\n=== Liczba NULL w kolumnach cech i label ===")
    df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in feature_cols + ["label"]
    ]).show(truncate=False)

    before = df.count()
    df = df.na.drop(subset=feature_cols + ["label"])
    after = df.count()
    print(f"\nUsunięto wiersze z brakami: {before - after} (pozostało {after})")

    return df, feature_cols


def compute_class_weights(df):
    total = df.count()
    dist = (
        df.groupBy("label")
          .agg(F.count("*").alias("count"))
          .orderBy("label")
    )

    print("\n=== Rozkład klas (label) ===")
    dist.show()

    counts = {row["label"]: row["count"] for row in dist.collect()}
    n0 = counts.get(0, 1)
    n1 = counts.get(1, 1)

    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)

    print(f"\nWagi klas: label=0 -> {w0:.4f}, label=1 -> {w1:.4f}")

    df = df.withColumn(
        "classWeightCol",
        F.when(F.col("label") == 0, F.lit(w0)).otherwise(F.lit(w1))
    )

    return df


def build_pipeline(feature_cols):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeightCol",
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0,
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])
    return pipeline


def evaluate_model(pred_df, dataset_name: str):
    print(f"\n=== Ewaluacja na zbiorze: {dataset_name} ===")

    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    prec_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    rec_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    accuracy = acc_eval.evaluate(pred_df)
    f1 = f1_eval.evaluate(pred_df)
    precision = prec_eval.evaluate(pred_df)
    recall = rec_eval.evaluate(pred_df)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    # confusion matrix
    cm_df = (
        pred_df.groupBy("label", "prediction")
               .agg(F.count("*").alias("count"))
               .orderBy("label", "prediction")
    )
    cm_pdf = cm_df.toPandas()

    # budujemy 2x2 macierz (zakładamy etykiety 0/1)
    cm = np.zeros((2, 2), dtype=int)
    for _, row in cm_pdf.iterrows():
        true_lbl = int(row["label"])
        pred_lbl = int(row["prediction"])
        cm[true_lbl, pred_lbl] = int(row["count"])

    print("\nMacierz pomyłek (label vs prediction):")
    print(cm)

    # zapis wykresu + csv
    plot_and_save_confusion_matrix(cm, labels=["0", "1"], dataset_name=dataset_name)

    metrics = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "f1": f1,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "cm_00": int(cm[0, 0]),
        "cm_01": int(cm[0, 1]),
        "cm_10": int(cm[1, 0]),
        "cm_11": int(cm[1, 1]),
    }
    save_metrics(metrics, dataset_name)

    return metrics


def export_lr_curves_and_coeffs(model: PipelineModel):
    """
    Dodatkowe rzeczy pod raport:
    - krzywa ROC i PR (dla treningu)
    - współczynniki logistycznej regresji
    """
    ensure_dir(REPORT_DIR)

    # ostatni stage to LogisticRegressionModel
    lr_model = model.stages[-1]
    summary = lr_model.summary  # BinaryLogisticRegressionTrainingSummary

    # ROC
    roc_df = summary.roc.toPandas()
    plot_and_save_curve(
        roc_df,
        x_col="FPR",
        y_col="TPR",
        title="ROC curve (train)",
        filename="roc_curve_train.png"
    )

    # PR
    pr_df = summary.pr.toPandas()
    plot_and_save_curve(
        pr_df,
        x_col="recall",
        y_col="precision",
        title="Precision-Recall curve (train)",
        filename="pr_curve_train.png"
    )

    # współczynniki
    coeffs = lr_model.coefficients.toArray()
    intercept = lr_model.intercept

    coef_df = pd.DataFrame({
        "feature": ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"],
        "coefficient": coeffs
    })
    coef_df["abs_coeff"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coeff", ascending=False)

    coef_path = os.path.join(REPORT_DIR, "lr_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"Zapisano współczynniki LR do: {coef_path}")

    # prosty wykres słupkowy współczynników
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(coef_df["feature"], coef_df["coefficient"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Współczynniki Logistic Regression")
    ax.set_ylabel("coefficient")
    ax.set_xlabel("feature")
    ax.grid(axis="y")

    coef_plot_path = os.path.join(REPORT_DIR, "lr_coefficients.png")
    plt.savefig(coef_plot_path)
    plt.close(fig)
    print(f"Zapisano wykres współczynników LR do: {coef_plot_path}")


def main():
    spark = create_spark()

    # 1. Dane
    df, feature_cols = load_data(spark)

    # 2. Wagi klas
    df = compute_class_weights(df)

    # 3. Podział na train/test
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    print(f"\nPodział danych: train={train_df.count()}, test={test_df.count()}")

    # 4. Pipeline i trening
    pipeline = build_pipeline(feature_cols)

    print("\n=== Trening modelu (Logistic Regression) ===")
    model = pipeline.fit(train_df)

    # 5. Ewaluacja + zapisywanie wykresów
    train_pred = model.transform(train_df)
    test_pred = model.transform(test_df)

    evaluate_model(train_pred, "train")
    evaluate_model(test_pred, "test")

    # 6. Dodatkowe rzeczy z LR (ROC/PR/coeff)
    export_lr_curves_and_coeffs(model)

    # 7. Zapis modelu
    print(f"\nZapisuję model do: {MODEL_PATH}")
    model.write().overwrite().save(MODEL_PATH)
    print("\nModel zapisany. Można go użyć w streamingu / backendzie.")

    spark.stop()


if __name__ == "__main__":
    main()
