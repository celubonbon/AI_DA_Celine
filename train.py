"""Entrena y evalúa un clasificador de hits musicales.

El objetivo es predecir ``hit = 1`` cuando ``Popularity >= 70`` usando las
características de audio disponibles. Se excluyen artistas y nombres de
canciones como predictores; el artista se reserva para separar los grupos y
evitar que canciones del mismo artista aparezcan en train y test.

El flujo imprime el perfil del CSV, ajusta un umbral de probabilidad usando
solo una validación separada por artista dentro del train, compara un baseline
de clase mayoritaria con RandomForest, calcula importancia por permutación y
guarda el modelo seleccionado con sus metadatos en formato joblib.

Uso desde la raíz del proyecto:
    python train.py
    python train.py --data data/train.csv --model-out models/best_model.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier


TARGET = "Popularity"
GROUP_COLUMN = "Artist Name"
# Solo se usan atributos de audio; se excluyen la etiqueta y datos de identidad.
FEATURE_CANDIDATES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_in min/ms",
    "time_signature",
]


def profile_dataset(data: pd.DataFrame) -> None:
    """Imprime esquema, faltantes, duplicados y estadísticas descriptivas."""
    print(f"Dataset: {data.shape[0]} filas x {data.shape[1]} columnas")
    print("\nColumnas y tipos:")
    print(data.dtypes.to_string())
    print("\nValores nulos por columna:")
    print(data.isna().sum().to_string())
    print(f"\nFilas duplicadas: {data.duplicated().sum()}")
    print("\nEstadísticas descriptivas:")
    print(data.describe(include="all").transpose().to_string())


def make_model(feature_data: pd.DataFrame) -> Pipeline:
    """Crea un pipeline que aprende imputación/codificación solo del ajuste."""
    numeric_features = feature_data.select_dtypes(include="number").columns.tolist()
    categorical_features = feature_data.select_dtypes(
        exclude="number"
    ).columns.tolist()
    transformers: list[tuple[str, Pipeline | SimpleImputer, list[str]]] = []

    if numeric_features:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_features)
        )
    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessing = ColumnTransformer(transformers=transformers)
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [("preprocessing", preprocessing), ("classifier", classifier)]
    )


def main() -> None:
    """Carga los datos, entrena, compara y persiste el clasificador."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/train.csv"),
        help="Ruta al CSV etiquetado (por defecto: data/train.csv)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/best_model.joblib"),
        help="Ruta donde guardar el modelo entrenado",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    profile_dataset(data)

    required = {TARGET, GROUP_COLUMN, *FEATURE_CANDIDATES}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    # Una etiqueta faltante no permite construir el objetivo supervisado.
    data = data.drop_duplicates().copy()
    data = data.dropna(subset=[TARGET]).copy()
    data["hit"] = data[TARGET].ge(70).astype("int8")

    features = data[FEATURE_CANDIDATES].copy()
    groups = data[GROUP_COLUMN].fillna("__ARTISTA_DESCONOCIDO__").astype(str)
    labels = data["hit"]

    # Artist Name define grupos, pero no se entrega al clasificador.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(
        splitter.split(features, labels, groups=groups)
    )
    x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = labels.iloc[train_indices], labels.iloc[test_indices]
    train_artists = set(groups.iloc[train_indices])
    test_artists = set(groups.iloc[test_indices])
    if train_artists.intersection(test_artists):
        raise RuntimeError("La partición contiene artistas en ambos conjuntos.")
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Train y test deben contener ambas clases para evaluar AUC.")

    # Reservar validación por artista permite ajustar umbral sin tocar el test.
    validation_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=43
    )
    fit_indices, validation_indices = next(
        validation_splitter.split(
            x_train, y_train, groups=groups.iloc[train_indices]
        )
    )
    threshold_model = make_model(x_train.iloc[fit_indices])
    threshold_model.fit(
        x_train.iloc[fit_indices], y_train.iloc[fit_indices]
    )
    validation_probabilities = threshold_model.predict_proba(
        x_train.iloc[validation_indices]
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_train.iloc[validation_indices], validation_probabilities
    )
    validation_f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    selected_threshold = float(thresholds[int(np.argmax(validation_f1))])
    print(
        f"\nUmbral seleccionado con validación por artista: "
        f"{selected_threshold:.4f} (F1 validación={validation_f1.max():.4f})"
    )

    models = {
        "Baseline (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "RandomForest": make_model(x_train),
    }
    results: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        threshold = selected_threshold if name == "RandomForest" else 0.5
        prediction = (probability >= threshold).astype("int8")
        predictions[name] = prediction
        results.append(
            {
                "Modelo": name,
                "F1": f1_score(y_test, prediction, zero_division=0),
                "AUC-ROC": roc_auc_score(y_test, probability),
                "Accuracy": accuracy_score(y_test, prediction),
            }
        )

    results_table = pd.DataFrame(results).set_index("Modelo")
    print(
        f"\nTarget: hit = Popularity >= 70 | filas utilizadas: {len(data)}"
        f" | hits: {int(labels.sum())} ({labels.mean():.1%})"
    )
    print(
        f"Train: {len(train_indices)} filas / {len(train_artists)} artistas; "
        f"test: {len(test_indices)} filas / {len(test_artists)} artistas"
    )
    print(f"Artistas compartidos entre train/test: {len(train_artists & test_artists)}")
    print("\nComparación de métricas:")
    print(results_table.round(4).to_string())
    for name, prediction in predictions.items():
        print(
            f"\nMatriz de confusión — {name} "
            "(filas=real [0,1], columnas=predicción [0,1]):"
        )
        print(confusion_matrix(y_test, prediction, labels=[0, 1]))

    # F1 prioriza recuperar la clase minoritaria; AUC resuelve empates.
    best_name = max(
        results,
        key=lambda result: (result["F1"], result["AUC-ROC"]),
    )["Modelo"]
    best_model = models[best_name]
    print(f"\nModelo seleccionado por F1 (desempate por AUC-ROC): {best_name}")

    if best_name == "RandomForest":
        importance = permutation_importance(
            best_model,
            x_test,
            y_test,
            scoring=lambda estimator, x, y: f1_score(
                y,
                estimator.predict_proba(x)[:, 1] >= selected_threshold,
                zero_division=0,
            ),
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        importance_table = pd.DataFrame(
            {
                "Variable": FEATURE_CANDIDATES,
                "Importancia media (caída F1)": importance.importances_mean,
                "Desv. estándar": importance.importances_std,
            }
        ).sort_values("Importancia media (caída F1)", ascending=False)
        print("\nImportancia por permutación (F1):")
        print(importance_table.to_string(index=False, float_format="%.5f"))
    else:
        print(
            "\nEl baseline fue el mejor; no se calcula importancia de variables "
            "para el modelo constante."
        )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "features": FEATURE_CANDIDATES,
            "target": "hit (Popularity >= 70)",
            "threshold": 70,
            "probability_threshold": (
                selected_threshold if best_name == "RandomForest" else 0.5
            ),
            "selected_model": best_name,
        },
        args.model_out,
    )
    print(f"\nModelo guardado en: {args.model_out}")


if __name__ == "__main__":
    main()
