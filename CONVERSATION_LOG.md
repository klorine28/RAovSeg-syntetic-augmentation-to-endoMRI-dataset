# Replicación del Pipeline RAovSeg — Conversación Cronológica

**Fecha**: 15 de abril de 2026

---

## Turno 1 — Usuario

> Quisiera replicar el pipeline de segmentación de las imágenes que describen en el paper s41597-025-05623-3. Plantea una estrategia para lograrlo.

---

## Turno 1 — Asistente

Se investigó el paper y el estado del proyecto en paralelo.

### Hallazgos sobre el paper

**Título**: *"A Multi-Modal Pelvic MRI Dataset for Deep Learning-Based Pelvic Organ Segmentation in Endometriosis"*
- Autores: Xiaomin Liang, Linda A. Alpuing Radilla, Kamand Khalaj, et al. (UT Health Houston, Baylor College of Medicine)
- Publicación: Scientific Data, volume 12, article 1292 (julio 2025)

**Datasets**:
- D1_MHS: 51 pacientes, multicéntrico, 15 sitios, 9 scanners (GE/Philips/Siemens), 1.5T y 3T
- D2_TCPW: 81 pacientes, monocéntrico (Texas Children's Hospital), Philips Ingenia 1.5T

**Pipeline RAovSeg (dos etapas)**:

| Etapa | Modelo | Arquitectura | Loss | Datos |
|-------|--------|-------------|------|-------|
| Stage 1: ResClass | ResNet18 2 capas (MONAI) | Features [8, 16] | BCEWithLogitsLoss | 3,252 train / 2,168 val slices |
| Stage 2: AttUSeg | Attention U-Net 4 capas (MONAI) | Features [16, 32, 64, 128] | Focal Tversky (α=0.8, β=0.2, γ=1.33) | 594 train / 136 val slices |

**Preprocesamiento**: Resample 512×512, spacing (0.35, 0.35, 6.0), percentile clip 1-99, min-max norm, realce custom (o1=0.24, o2=0.3)

**Post-procesamiento**: Cierre morfológico (10 iter) + componente conectado más grande

**Resultados**: Full pipeline DSC=0.290, sin post-proc DSC=0.235, sin ResClass DSC=0.013, nnU-Net DSC=0.272

**Gaps**: LR, batch size, epochs, optimizer no reportados en el paper.

**Repos**: Dataset en Zenodo (zenodo.org/records/13749613), código en GitHub (github.com/xlianguth/RAovSeg)

### Estado del proyecto

Solo contenía el folder `Paper/` con el PDF. Sin código, sin datos, sin estructura.

### Estrategia propuesta

- **Fase 0**: Setup del proyecto, descargar datos de Zenodo, clonar código, crear ambiente
- **Fase 1**: Preprocesamiento (resample, normalize, enhance)
- **Fase 2**: Entrenar ResClass (clasificador de slices)
- **Fase 3**: Entrenar AttUSeg (segmentador) + post-procesamiento
- **Fase 4**: Evaluación + comparación con benchmarks + nnU-Net

Se incluyó un plan por semanas como sugerencia de timeline.

---

## Turno 2 — Usuario

> ¿Por qué sugieres que sea por semanas?

---

## Turno 2 — Asistente

Se reconoció el error: no tiene sentido imponer una línea temporal arbitraria. El plan se reformuló como fases secuenciales sin estimaciones de tiempo. El ritmo lo define el usuario.

---

## Turno 3 — Usuario

> Sí, los datos ya están descargados en el folder UT-EndoMRI.

---

## Turno 3 — Asistente

### Exploración de datos

Se exploró la estructura de `UT-EndoMRI/`:

```
UT-EndoMRI/
├── D1_MHS/     (51 sujetos)
├── D2_TCPW/    (73 sujetos)
└── README.md
```

### Data split (del README oficial)

- **Train/Val**: D2-000 a D2-007 (8 sujetos)
- **Test**: D2-008 a D2-037 (30 sujetos)
- Solo Dataset 2 se usa para RAovSeg

### Análisis de sujetos train/val

| Sujeto | Secuencias | Label ovario | Quiste/Endometrioma |
|--------|-----------|-------------|---------------------|
| D2-000 | T1, T2, T2FS | ov | cy |
| D2-001 | T1, T1FS, T2, T2FS | ov | — |
| D2-002 | T1, T1FS, T2, T2FS | — | cy |
| D2-003 | T1, T1FS, T2, T2FS | ov | cy |
| D2-004 | T1, T1FS, T2, T2FS | — | — |
| D2-005 | T1, T1FS, T2, T2FS | ov | — |
| D2-006 | T1, T1FS, T2, T2FS | — | cy |
| D2-007 | T1, T1FS, T2, T2FS | ov | — |

**Excluidos del training** (tienen cy/em): D2-000, D2-002, D2-003, D2-006

**Sujetos usables para training**: D2-001, D2-005, D2-007 (con label de ovario, sin cy/em)

Ejemplo de archivos por sujeto (D1):
```
D1-000_T1FS.nii.gz, D1-000_T2.nii.gz, D1-000_em_r1.nii.gz, D1-000_em_r2.nii.gz,
D1-000_em_r3.nii.gz, D1-000_ov_r1.nii.gz, D1-000_ut_r1.nii.gz, ...
```

### Código de referencia clonado

Se clonó `https://github.com/xlianguth/RAovSeg` con:
- `RAovSeg_tools.py` — ImgResample, ImgNorm, preprocess_, postprocess_, dsc_cal_np, dsc_cal_torch
- `tutorial.py` — demo del pipeline
- Archivos ejemplo: UTEndoMRI_example.nii.gz, OvLabel.nii.gz, Prediction.nii.gz

### Scripts creados

Se crearon 4 scripts y 1 archivo de dependencias:

**`src/preprocess.py`**
- Procesa sujetos de D2_TCPW separando en train_val/, excluded/, test/
- Prioriza T2FS como secuencia de entrada
- Usa funciones de RAovSeg_tools (ImgResample, ImgNorm, preprocess_)
- Guarda arrays .npy (image.npy + ov_label.npy)
- Parámetros: spacing (0.35, 0.35, 6.0), size 512×512×38, percentile 1-99, o1=0.24, o2=0.3

**`src/train_resclass.py`**
- ResNet18 (MONAI), dropout 0.2, BCEWithLogitsLoss
- Augmentation: rotaciones ±25°, traslaciones ±25px, multiplicador 5x
- Adam lr=1e-3, weight_decay=1e-4, ReduceLROnPlateau
- Train/val split 60/40, guarda mejor modelo por val accuracy

**`src/train_attuseg.py`**
- Attention U-Net (MONAI), features [16, 32, 64, 128]
- Focal Tversky Loss custom (α=0.8, β=0.2, γ=1.33)
- Solo entrena con slices que contienen ovario
- Misma augmentation y optimizer config que ResClass

**`src/evaluate.py`**
- Pipeline completo: ResClass → AttUSeg → post-processing → DSC
- Incluye ablaciones (sin post-processing, sin ResClass)
- Compara contra benchmarks del paper

**`requirements.txt`**: numpy, torch, monai, SimpleITK, scipy, matplotlib

### Estructura final del proyecto

```
EndometriosisDataset/
├── UT-EndoMRI/            # Datos descargados
├── RAovSeg/               # Código de referencia clonado
├── src/
│   ├── preprocess.py      # Fase 1
│   ├── train_resclass.py  # Fase 2
│   ├── train_attuseg.py   # Fase 3
│   └── evaluate.py        # Fase 4
├── data/processed/        # Output del preprocesamiento
├── models/                # Checkpoints
└── requirements.txt
```

### Orden de ejecución

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train_resclass.py
python src/train_attuseg.py
python src/evaluate.py
```

Se preguntó al usuario si quería ejecutar el preprocesamiento.

---

## Turno 4 — Usuario

> Crea un archivo md con la conversación desde el inicio.

---

## Turno 4 — Asistente

Se creó `CONVERSATION_LOG.md` organizado por temas.

---

## Turno 5 — Usuario

> Puedes hacer uno con la conversación en orden cronológico.

---

## Turno 5 — Asistente

Se reescribió este archivo en formato cronológico.

---

## Referencias

- **Paper**: https://doi.org/10.1038/s41597-025-05623-3
- **Dataset (Zenodo)**: https://zenodo.org/records/13749613
- **Código (GitHub)**: https://github.com/xlianguth/RAovSeg
- **Licencia datos**: Non-commercial scientific research only
- **Licencia código**: MIT
