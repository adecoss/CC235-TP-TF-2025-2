# CC235 – Trabajo Parcial/Final 2025-2

## Objetivo del Trabajo

El objetivo del presente proyecto es desarrollar un **sistema de clasificación automática de estilos arquitectónicos a partir de imágenes**. Se busca aplicar técnicas de **procesamiento de imágenes y aprendizaje profundo**, comparando modelos clásicos, CNNs modernas y transformadores visuales para identificar estilos arquitectónicos (Art Decó, Gótico, Barroco, Bauhaus, etc.).  

El proyecto permite extraer conocimiento útil para:  

- Catalogación histórica y conservación del patrimonio.  
- Búsqueda y recomendación de bienes raíces por estilo arquitectónico.  
- Desarrollo de herramientas de diseño asistido para arquitectos.  

---

## Alumnos Participantes

- Colfer Mendoza Carlos Alejandro - U20241B820
- De Cossio Velasquez Alvaro Manuel - U20221F812
- Pacherres Muñoz Peter Smith - U202423854
---

## Descripción del Dataset

Se utilizó el **Architecture Dataset de Kaggle (Mak, Wendy, 2018)**, compuesto por más de **4 700 imágenes** de edificios etiquetadas en **25 estilos arquitectónicos diferentes**.  
https://www.kaggle.com/datasets/wwymak/architecture-dataset/data 

### Procesamiento del Dataset
- Se realizó un **split inicial** en **entrenamiento (80%) y validación (20%)**.  
- Las imágenes fueron **redimensionadas a 224×224 píxeles** y normalizadas para los modelos preentrenados.  
- Se aplicaron técnicas de **augmentación**, incluyendo RandAugment y flips horizontales, para mejorar la generalización.  

### Carpetas en el Repositorio
- `data/`  
  - Contiene el **dataset original**.  
  - Contiene el **dataset final preparado**, listo para entrenamiento y evaluación.  
- `code/`  
  - Contiene los **programas en Python** desarrollados para entrenamiento, validación y comparación de modelos.

---

## Modelos Evaluados

Se entrenaron y compararon los siguientes modelos:  

- **CNN Básica:** Red convolucional simple como línea base.  
- **ConvNeXt-Small:** CNN moderna optimizada, versión pequeña.  
- **EfficientNetV2-RW-S:** CNN escalable y eficiente, versión pequeña.  
- **ResNet-RS50:** Variante mejorada de ResNet, con anti-aliasing y optimización de entrenamiento.  
- **Swin Small:** Transformer jerárquico con ventanas deslizantes.  
- **DEiT III:** Vision Transformer con distillation preentrenado.  
- **TinyViT 21M:** Transformer ligero con convoluciones locales y autoatención global.  

### Técnicas Aplicadas
- **Mixup / CutMix:** Data augmentation avanzada que combina imágenes y etiquetas para mejorar generalización.  
- **Label Smoothing:** Suavizado de etiquetas para evitar sobreconfianza en la predicción.  
- **AdamW + CosineLRScheduler:** Optimización estable con decaimiento de pesos desacoplado y curva coseno de learning rate.  
- **Early Stopping (7 epochs):** Para evitar sobreajuste durante el entrenamiento.  

### Parámetros Base
- **EARLY_STOP_PATIENCE:** 7  
- **BATCH_SIZE:** 64  
- **IMG_SIZE:** 224  
- **NUM_CLASSES:** 25  
- **EPOCHS:** 50  
- **LR:** 3e-4  

**Hardware:** Entrenamiento en **RTX 4060 personal** y **RTX 3070 de la universidad**.

---

## Conclusiones

- **TinyViT 21M** resultó ser el modelo con **mejor desempeño**, combinando precisión, eficiencia y velocidad.  
- Técnicas de **augmentación y regularización** mejoran significativamente la generalización.  
- Arquitecturas modernas basadas en **transformers visuales** superan a CNNs clásicas en clasificación de estilos arquitectónicos.  
- Se establece una **base reproducible** para proyectos futuros de análisis de imágenes arquitectónicas.  

---

## Licencia

Este proyecto está licenciado bajo **[MIT License]**. Se permite su uso académico y personal, siempre citando este trabajo.

