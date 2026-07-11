# Flujograma — Segunda Exposición Académica MCC225

Este documento traza, con diagramas Mermaid como fuente de verdad, el flujo completo
del avance integrador presentado para la **Segunda Exposición Académica** de
MCC225A (IA Generativa y Aprendizaje Multimodal). El proyecto integrador es la
evaluación de un **dual-encoder CLIP** sobre **Winoground**, extendida con el
`Cuaderno14` y la `Actividad5` (evaluación responsable).

Todos los números citados provienen de archivos reales del repositorio
(`outputs/metrics/`, `outputs/tables/`) y son reproducibles con `make avance`.

## 🔭 Flujo general del trabajo

_Flujograma de extremo a extremo: desde los insumos del curso hasta los entregables
evaluables, pasando por la ejecución del cuaderno y la evaluación responsable._

```mermaid
flowchart TB
    accTitle: Flujo general de la Exposicion 2
    accDescr: Los insumos del curso (Cuaderno 14 y Actividad 5) se sincronizan al repo, se construye un manifiesto local desde Winoground, se ejecuta el cuaderno y se evalua el sistema; la evidencia alimenta la Actividad 5, las diapositivas y el avance tecnico, todo versionado en el repositorio.

    subgraph insumos["📥 Insumos del curso"]
        consigna["📄 Consigna Exposicion 2"]
        cuaderno["📓 Cuaderno14 (base)"]
        actividad["📝 Actividad5 (consigna)"]
    end

    subgraph datos["🗂️ Datos y modelos"]
        manifest["🖼️ manifest_local.csv<br/>120 pares reales Winoground"]
        modelos["🧠 CLIP + BLIP<br/>safetensors locales"]
    end

    subgraph ejecucion["⚙️ Ejecucion reproducible"]
        c14run["🚀 Cuaderno14_resuelto<br/>retrieval · CLIPScore · perturbaciones"]
        winorun["📊 Pipeline Winoground<br/>text/image/group · bootstrap"]
    end

    subgraph evidencia["📈 Evidencia"]
        metrics["📑 outputs/metrics + tables"]
        figs["🖼️ outputs/figures"]
        errores["🔎 analisis de errores anotado"]
    end

    subgraph entregables["🎓 Entregables evaluables"]
        slides["🖥️ 8 diapositivas (PDF)"]
        avance["📃 Avance tecnico (2-3 pp)"]
        reporte["🧭 Reporte Actividad5"]
        repo["🗃️ Repositorio con commits"]
    end

    consigna --> c14run
    cuaderno --> c14run
    actividad --> reporte
    manifest --> c14run
    modelos --> c14run
    c14run --> metrics
    c14run --> figs
    c14run --> errores
    winorun --> metrics
    winorun --> figs
    metrics --> reporte
    errores --> reporte
    metrics --> slides
    figs --> slides
    metrics --> avance
    reporte --> slides
    slides --> repo
    avance --> repo
    reporte --> repo
    metrics --> repo

    classDef inputs fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef proc fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef data fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef out fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class consigna,cuaderno,actividad inputs
    class manifest,modelos data
    class c14run,winorun proc
    class metrics,figs,errores data
    class slides,avance,reporte,repo out
```

## 🧩 Arquitectura del modelo (dual-encoder CLIP)

_Diagrama de la arquitectura evaluada: dos torres transformer independientes (imagen
y texto) con self-attention interna, unidas solo al final por similitud coseno. La
ausencia de cross-attention entre torres es la razon mecanistica de la composicion
debil._

```mermaid
flowchart LR
    accTitle: Arquitectura dual-encoder CLIP
    accDescr: La imagen pasa por un ViT con self-attention y el texto por un transformer con self-attention; ambas torres son independientes y solo se comparan por similitud coseno al final, sin cross-attention, lo que limita el razonamiento composicional.

    img["🖼️ Imagen"] --> patches["🧩 Parches + pos.<br/>tokens visuales"]
    patches --> vit["🏛️ ViT<br/>self-attention"]
    vit --> vemb["📍 Embedding imagen"]

    txt["📝 Caption"] --> tok["🔤 Tokens + pos."]
    tok --> ttf["🏛️ Text transformer<br/>self-attention"]
    ttf --> temb["📍 Embedding texto"]

    vemb --> cos["📐 Similitud coseno"]
    temb --> cos
    cos --> score["🎯 text / image / group score"]

    classDef mod fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef enc fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef join fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class img,txt,patches,tok mod
    class vit,ttf,vemb,temb enc
    class cos,score join
```

## ⚙️ Pipeline de evaluación del Cuaderno 14

_Secuencia de etapas del cuaderno resuelto: carga de datos reales, codificacion CLIP,
metricas de alineamiento, ablaciones de robustez y consolidacion de evidencia._

```mermaid
flowchart TB
    accTitle: Pipeline del Cuaderno 14
    accDescr: El manifiesto local alimenta la codificacion CLIP; de alli se derivan retrieval Recall@K, CLIPScore, captioning BLIP con BLEU/ROUGE, degradacion visual, perturbacion textual y una plantilla de analisis de errores; todo se consolida en metricas finales y figuras.

    m["🖼️ manifest_local.csv"] --> enc["🧠 Codificacion CLIP<br/>imagen + texto"]
    enc --> ret["📊 Recall@K (i2t / t2i)"]
    enc --> cs["📐 CLIPScore diagonal"]
    enc --> blip["🖊️ Captioning BLIP<br/>BLEU · ROUGE-L · cobertura"]
    enc --> deg["🌫️ Degradacion visual<br/>blur · gris · baja-res · recorte"]
    enc --> pert["🔀 Perturbacion textual<br/>negacion · conteo · atributo"]
    cs --> err["🔎 Plantilla de errores<br/>10 categorias, anotada"]
    ret --> fin["📑 metricas_finales.json"]
    cs --> fin
    blip --> fin
    deg --> fin
    pert --> fin
    fin --> rep["🧾 reporte_exposicion_2.md"]

    classDef data fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef proc fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef out fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class m data
    class enc,ret,cs,blip,deg,pert,err proc
    class fin,rep out
```

## 🗺️ Trazabilidad rúbrica → archivo

_Mapa de cada criterio de la rúbrica (20 pts) al artefacto del repositorio que lo
sustenta. Sirve como checklist de defensa._

```mermaid
flowchart LR
    accTitle: Trazabilidad rubrica a archivos
    accDescr: Cada criterio de la rubrica se enlaza al archivo del repositorio que aporta su evidencia, desde formulacion hasta comunicacion.

    r1["🎯 Formulacion (3)"] --> a1["📃 avance_tecnico · slide 2"]
    r2["🏛️ Transformer + atencion (4)"] --> a2["🖥️ slide 4 · avance §3"]
    r3["🔀 Arquitecturas (4)"] --> a3["🗂️ ADR 0001 · slide 3"]
    r4["📊 Evidencia cuaderno (3)"] --> a4["📓 Cuaderno14_resuelto · outputs"]
    r5["🧭 Actividad5 (3)"] --> a5["📝 evaluacion_responsable_mcc225"]
    r6["🗃️ Repo/repro (2)"] --> a6["🐳 Docker · Makefile · CI"]
    r7["🗣️ Comunicacion (1)"] --> a7["💬 RESPUESTAS_PREGUNTAS"]

    classDef crit fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef file fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class r1,r2,r3,r4,r5,r6,r7 crit
    class a1,a2,a3,a4,a5,a6,a7 file
```

## 🐳 Reproducción

Dos caminos equivalentes; ambos regeneran datos, modelos, evidencia y figuras.

| Camino | Comando |
|---|---|
| Entorno local (uv/venv) | `make setup && make models && make avance && make run && make figures && make test` |
| Docker | `docker compose run --rm avance` |

Documentos relacionados: [PLAN_EXPOSICION2_MCC225.md](PLAN_EXPOSICION2_MCC225.md) ·
[ENTREGA_EXPOSICION2.md](ENTREGA_EXPOSICION2.md) ·
[adr/0002-cuaderno14-sobre-winoground.md](adr/0002-cuaderno14-sobre-winoground.md) ·
[../Actividad5-MCC225.md](../Actividad5-MCC225.md)
