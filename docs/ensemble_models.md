# Modelos ensemble y feature importance

Este documento describe el trabajo implementado para Random Forest, XGBoost,
su evaluación común y la interpretación de sus variables. Los resultados
mostrados durante el desarrollo proceden de pruebas sintéticas y no deben
confundirse con los resultados finales del dataset de reservas.

## Estado de implementación

| Componente | Estado |
| --- | --- |
| Random Forest base | Implementado y probado con datos sintéticos |
| XGBoost base | Implementado y probado con datos sintéticos |
| Métricas y visualizaciones comunes | Implementadas |
| Tabla y ROC comparativas | Implementadas |
| Selección por métrica configurable | Implementada, pendiente de integración final |
| Feature importance genérica | Implementada y probada con ambos ensembles |
| Entrenamiento definitivo | Pendiente del preprocesamiento clásico |
| Importancias y métricas reales | Pendientes |
| Selección del modelo ganador | Pendiente |

## Contrato común de datos

Una comparación justa exige que todos los modelos partan del mismo split. El
preprocesamiento de los modelos clásicos y el de la red neuronal puede ser
distinto, pero ambos deben recibir las mismas observaciones base de train y
test, en el mismo orden.

El contrato esperado para los modelos clásicos es:

```python
X_train_preprocessed, X_test_preprocessed, y_train, y_test
```

Random Forest y XGBoost no realizan su propio `train_test_split()`. Ambos
reciben `X_train_preprocessed` e `y_train` desde el flujo común y deben
evaluarse con el mismo `X_test_preprocessed` e `y_test`.

El preprocesador clásico deberá cumplir estas condiciones:

- Ajustar imputadores y codificadores únicamente con train.
- Transformar test sin volver a ajustar ningún componente.
- Convertir las variables categóricas a una representación numérica.
- Mantener la correspondencia entre cada fila y su target.
- No balancear artificialmente el conjunto de test.
- Exponer los nombres transformados en el mismo orden que las columnas.

## Random Forest

Random Forest crea múltiples árboles de decisión a partir de muestras y
subconjuntos de variables aleatorios. Los árboles se entrenan de forma
independiente y sus resultados se combinan para producir la predicción final.
Esta combinación suele reducir la varianza y el sobreajuste respecto a un
único árbol.

La configuración base del proyecto es:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
```

- `n_estimators=100` crea cien árboles como punto de partida.
- `random_state=42` permite reproducir el entrenamiento.
- `n_jobs=-1` utiliza los núcleos disponibles para entrenar.
- `class_weight` conserva su valor por defecto. No se aplica balanceo sin
  comprobar antes si es necesario.

Es una configuración base, no el resultado de una optimización de
hiperparámetros. Cualquier ajuste posterior deberá utilizar validación sobre
train, nunca el conjunto de test.

## XGBoost

XGBoost construye árboles de forma secuencial. Cada nuevo árbol intenta
corregir parte de los errores cometidos por el conjunto anterior. A diferencia
de Random Forest, sus árboles no son independientes y la contribución de cada
nuevo árbol está controlada por una tasa de aprendizaje.

La configuración base del proyecto es:

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
)
```

- `binary:logistic` adapta el modelo a clasificación binaria y permite obtener
  probabilidades.
- `logloss` evalúa internamente la calidad de las probabilidades durante el
  entrenamiento. No determina por sí sola el ganador final.
- `n_estimators=100` establece cien rondas de boosting.
- `learning_rate=0.1` limita la aportación de cada árbol.
- `max_depth=6` controla la complejidad de los árboles.
- `random_state=42` permite reproducir el resultado.
- `n_jobs=-1` utiliza los núcleos disponibles.

No se ha configurado `scale_pos_weight`, porque el tratamiento del desbalance
debe justificarse primero. Tampoco se ha incorporado early stopping: hacerlo
correctamente requiere un conjunto de validación separado dentro de train.

## Comparación entre ambos modelos

| Aspecto | Random Forest | XGBoost |
| --- | --- | --- |
| Construcción | Árboles independientes | Árboles secuenciales |
| Objetivo principal | Reducir varianza mediante agregación | Corregir progresivamente los errores |
| Parámetro característico | Número de árboles | Número de rondas y learning rate |
| Ajuste | Generalmente más sencillo | Más sensible a hiperparámetros |
| Probabilidades | `predict_proba()` | `predict_proba()` |

Ninguno debe declararse ganador por su diseño teórico. La decisión debe basarse
en métricas calculadas sobre las mismas observaciones de test.

## Evaluación común

Los dos ensembles utilizan las mismas definiciones de:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC calculado con scores de la clase positiva.

El evaluador también permite generar matrices de confusión, curvas ROC
individuales, una ROC comparativa y una tabla común. La selección del mejor
modelo admite `accuracy`, `precision`, `recall`, `f1` o `roc_auc` como métrica
principal.

El uso provisional de F1 en `model_trainer.py` no constituye todavía una
selección definitiva. Antes de decidir la métrica principal deben revisarse el
balance de clases, el coste de falsos positivos y falsos negativos y los
resultados de todos los modelos.

## Feature importance

La utilidad genérica funciona con modelos entrenados que expongan el atributo
`feature_importances_`. Asocia cada valor con el nombre de la columna utilizada
durante el entrenamiento, genera una tabla ordenada y permite representar las
variables más importantes mediante `top_n`.

La integración futura tendrá esta forma:

```python
from src.feature_importance import (
    create_feature_importance_table,
    plot_feature_importance,
)

feature_names = classical_preprocessor.get_feature_names_out()

importance_table = create_feature_importance_table(
    random_forest_model,
    feature_names,
)

axis = plot_feature_importance(
    random_forest_model,
    feature_names,
    model_name="Random Forest",
    top_n=20,
)
```

El ejemplo representa el contrato esperado; no implica que
`classical_preprocessor` esté disponible actualmente.

La posición debe coincidir exactamente:

```text
model.feature_importances_[i] <-> feature_names[i]
```

No se pueden inventar los nombres a partir de una matriz transformada. El
preprocesador clásico deberá proporcionar `get_feature_names_out()` o una
interfaz equivalente.

### Limitaciones de la interpretación

- Feature importance establece asociaciones predictivas, no causalidad.
- Las importancias de árboles pueden favorecer variables continuas o de alta
  cardinalidad.
- Las categorías generadas por One-Hot Encoding aparecen inicialmente como
  variables separadas.
- Random Forest y XGBoost pueden calcular la importancia con criterios
  diferentes; sus valores no deben compararse directamente entre modelos.
- La utilidad no redondea, renormaliza ni modifica los valores del estimador.

## Integración y resultados pendientes

Cuando esté disponible el preprocesamiento clásico será necesario:

1. Confirmar que todos los modelos usan el mismo split base.
2. Entrenar los ensembles con los datos reales de train.
3. Obtener los nombres transformados en el orden correcto.
4. Evaluar todos los modelos sobre el mismo test.
5. Generar las tablas, gráficas e importancias reales.
6. Analizar el desbalance antes de aplicar cualquier corrección.
7. Elegir y justificar la métrica principal.
8. Seleccionar el ganador únicamente cuando estén todos los resultados.

Hasta entonces no deben publicarse métricas, importancias ni un modelo ganador
como resultados finales.

## Conceptos para la defensa

- Un ensemble combina varios modelos para obtener una predicción más robusta.
- Random Forest agrega árboles independientes; XGBoost construye árboles que
  corrigen secuencialmente los errores anteriores.
- La semilla aleatoria facilita la reproducibilidad, pero no garantiza por sí
  sola una metodología correcta.
- Una comparación justa requiere las mismas observaciones de test y las mismas
  definiciones de métricas.
- El test no se utiliza para ajustar preprocesadores, balancear, entrenar ni
  elegir hiperparámetros.
- `predict()` produce clases; `predict_proba()` proporciona los scores
  necesarios para ROC-AUC.
- Feature importance ayuda a interpretar el modelo, pero no demuestra una
  relación causal.
- La ausencia actual de resultados reales es una decisión metodológica: evita
  evaluar modelos sobre datos sin un preprocesamiento común y definitivo.
