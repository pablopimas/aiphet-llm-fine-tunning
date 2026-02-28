# Mejoras para `lab/aiprom-fhir-finetune.ipynb`

Análisis basado en el código real del notebook. Las mejoras están ordenadas por impacto descendente.

---

## ✅ 1. Eliminar duplicación de funciones entre celdas (impacto alto) — COMPLETADO

Las siguientes funciones están definidas en **múltiples celdas independientes**, generando inconsistencias silenciosas cuando se modifica una copia pero no las otras:

| Función | Celdas afectadas |
|---|---|
| `iter_questionnaire_items` | Celdas 9, 12, 18 |
| `_GoldRecord` (dataclass) | Celdas 8, 14 |
| `_resolve_root_dir` / `resolve_repo_root` | Celdas 8, 14, 17 |
| `_resolve_data_artifacts_dir` | Celdas 8, 14 |
| `slugify_model_name` | Celdas 2, 23 |

**Acción**: Mover todas estas funciones a la celda de _Shared Runtime Helpers_ (celda 17), que ya existe con ese propósito, y eliminarlas del resto de celdas. Añadir en las celdas dependientes un comentario `# Requires: run shared helpers cell first`.

---

## ✅ 2. Reemplazar el patrón `globals()` por un objeto de estado (impacto alto) — COMPLETADO

El notebook usa `if "variable" in globals()` en más de 30 lugares para resolver estado entre celdas:

```python
# Patrón actual (frágil)
if "artifacts_dir" in globals():
    return Path(artifacts_dir)
```

Esto es difícil de depurar, impide la reutilización fuera del notebook y falla silenciosamente si una celda no se ha ejecutado.

**Acción**: Crear un dataclass `NotebookState` en la celda de configuración que acumule el estado explícito del pipeline:

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class NotebookState:
    repo_root: Path = field(default_factory=Path.cwd)
    model_name: str = ""
    model_alias: str = ""
    artifacts_dir: Path = field(default_factory=Path)
    dataset_path: Path = field(default_factory=Path)
    parsed_records: list = field(default_factory=list)
    train_indices: list = field(default_factory=list)
    val_indices: list = field(default_factory=list)
    training_config: dict = field(default_factory=dict)

STATE = NotebookState()
```

Cada celda lee y escribe `STATE` en lugar de depender de variables globales sueltas.

---

## ✅ 3. Añadir visualizaciones con matplotlib/pandas (impacto medio-alto) — COMPLETADO

El notebook genera estadísticas relevantes pero solo las imprime como texto plano. Los datos ya están disponibles, solo falta renderizarlos.

**Celda de distribución del split** — añadir tras el resumen de texto:
```python
import pandas as pd
import matplotlib.pyplot as plt

df_dist = pd.DataFrame({
    "Train": train_distribution,
    "Val": val_distribution,
}).fillna(0).astype(int)

df_dist.plot(kind="bar", figsize=(10, 4), title="Distribución de tipos FHIR por split")
plt.ylabel("Nº de registros")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

**Celda de evaluación A/B** — añadir tabla comparativa:
```python
df_ab = pd.DataFrame({
    "Regla": list(baseline_rule_rates.keys()),
    "Baseline (%)": list(baseline_rule_rates.values()),
    "Adapter (%)": list(adapter_rule_rates.values()),
})
df_ab["Delta (pp)"] = df_ab["Adapter (%)"] - df_ab["Baseline (%)"]
df_ab.sort_values("Delta (pp)").style.background_gradient(subset=["Delta (pp)"], cmap="RdYlGn")
```

---

## ✅ 4. Unificar constantes de inferencia duplicadas (impacto medio) — COMPLETADO

La celda de A/B cualitativo (celda 32) y la celda de A/B cuantitativo (celda 34) definen sus propias constantes con valores distintos:

```python
# Celda 32
MAX_TOKENS = 768
SEED = 173

# Celda 34
EVAL_MAX_TOKENS = 900   # ← valor diferente
EVAL_SEED = 173
```

Un `MAX_TOKENS` diferente entre las dos evaluaciones puede producir resultados no comparables.

**Acción**: Mover todas las constantes de inferencia a la celda de configuración global (celda 2) y referenciarlas desde ambas celdas de evaluación:

```python
# En celda 2 (configuración global)
INFERENCE_MAX_TOKENS = 900
INFERENCE_SEED = 173
INFERENCE_TEMP = 0.0
INFERENCE_TOP_P = 1.0
```

---

## ✅ 5. Añadir split de test y smoke test post-fusión GGUF (impacto medio) — COMPLETADO

**Split de test**: el split actual es 80/20 train/val. No hay un conjunto de test que nunca haya influido en la selección de hiperparámetros ni en el criterio Go/No-Go.

**Acción**: Cambiar a 70/15/15 en `stratified_train_val_split` y añadir `test_indices` en `split_manifest.json`. Usar el test set solo en la evaluación final.

**Smoke test GGUF**: la celda de exportación GGUF (`RUN_GGUF_EXPORT`) no valida el artefacto generado. Una corrupción del proceso de fusión pasa desapercibida.

**Acción**: Añadir celda de verificación tras la exportación:

```python
# Smoke test: cargar GGUF con llama.cpp y verificar respuesta mínima
import subprocess, json

result = subprocess.run([
    str(REPO_ROOT / "tools" / "llama.cpp" / "build" / "bin" / "llama-cli"),
    "-m", str(gguf_path),
    "--prompt", "<|im_start|>user\nGenerate a minimal FHIR Questionnaire.<|im_end|>\n<|im_start|>assistant\n",
    "-n", "64",
    "--temp", "0",
    "--log-disable",
], capture_output=True, text=True, cwd=str(REPO_ROOT))

output = result.stdout.strip()
assert "{" in output and "resourceType" in output, f"Smoke test failed: {output[:200]}"
print("GGUF smoke test: OK")
```

---

## ✅ 6. Nombre del notebook vs modelo por defecto (impacto bajo-medio) — COMPLETADO

Renombrado de `qwen-7b.ipynb` a `aiprom-fhir-finetune.ipynb`, reflejando que el flujo es model-agnostic.

**Opciones**:
- Renombrar a `aiprom-finetune.ipynb` (refleja que es model-agnostic).
- Actualizar `DEFAULT_PUBLIC_MLX_MODEL` a un modelo 7B real (e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit`).

---

## ✅ 7. Eliminar `from __future__ import annotations` redundante (impacto bajo) — COMPLETADO

Aparece al inicio de prácticamente todas las celdas de código. En Python 3.10+ (que es el target del proyecto dado el uso de MLX), es innecesario y añade ruido visual.

**Acción**: Eliminar de todas las celdas excepto la celda de imports global (celda 5), que ya lo incluye.

---

## ✅ 8. Integrar experiment tracking real o eliminar `wandb` del snapshot (impacto bajo) — COMPLETADO

`wandb` aparece en el snapshot de entorno de la celda 2 pero nunca se instancia ni configura. Su presencia genera confusión sobre si el tracking está activo.

**Opciones**:
- Añadir inicialización de un run de W&B con los parámetros de `training_config` y loguear las métricas del resultado A/B al finalizar.
- Eliminar `wandb` del snapshot de paquetes y del `requirements.txt` si no se va a usar.

---

## Resumen de prioridades

| # | Mejora | Impacto | Estado |
|---|---|---|---|
| 1 | Consolidar funciones duplicadas en helpers compartidos | Alto | ✅ Completado |
| 2 | Reemplazar `globals()` por `NotebookState` | Alto | ✅ Completado |
| 3 | Añadir visualizaciones matplotlib/pandas | Medio-Alto | ✅ Completado |
| 4 | Unificar constantes de inferencia | Medio | ✅ Completado |
| 5 | Split de test + smoke test GGUF | Medio | ✅ Completado |
| 6 | Corregir nombre del notebook | Bajo-Medio | ✅ Completado |
| 7 | Eliminar `from __future__` redundante | Bajo | ✅ Completado |
| 8 | Resolver uso de `wandb` | Bajo | ✅ Completado |