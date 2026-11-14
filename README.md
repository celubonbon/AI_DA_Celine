# AI_DA_Celine

Soy Celine y mi data set es de canciones.

## Predicción de hits

`train.py` clasifica una canción como hit cuando `Popularity >= 70`. Usa
características de audio, divide los datos por artista para evitar artistas
compartidos entre train y test, compara el modelo con un baseline y guarda el
mejor modelo en `models/best_model.joblib`.

En Windows, instala las dependencias en el entorno virtual y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

El script imprime el perfil del dataset, F1, AUC-ROC, matrices de confusión e
importancia por permutación. El umbral de probabilidad se ajusta con una
validación separada por artista dentro del conjunto de entrenamiento.

El análisis paso a paso, con explicación de variables y métricas, está en
[`notebooks/analisis_popularidad_canciones.ipynb`](./notebooks/analisis_popularidad_canciones.ipynb).