import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import transforms
import timm
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Comparador de Modelos Arq.",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>🏛️ Clasificación de Estilos Arquitectónicos</h1>
        <p>Comparación: SVM • Vision Transformer • ConvNeXt</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR CON INFORMACIÓN ---
with st.sidebar:
    st.header("ℹ️ Información del Sistema")
    st.markdown("""
    ### 🧠 Modelo Clásico
    - **Técnica**: ORB + BoVW + SVM
    - **Features**: 500 puntos clave
    - **Ventaja**: Rápido y ligero
    
    ### 🤖 Vision Transformer
    - **Arquitectura**: ViT-Base
    - **Ventaja**: Mayor precisión
    - **Técnica**: Attention mechanism
    
    ### ⚡ ConvNeXt Small
    - **Arquitectura**: CNN moderna
    - **Ventaja**: Balance velocidad/precisión
    - **Técnica**: Convoluciones optimizadas
    """)
    
    st.divider()
    
    st.markdown("""
    ### 📊 Métricas de Confianza
    - 🟢 **Alta**: > 80%
    - 🟡 **Media**: 60-80%
    - 🔴 **Baja**: < 60%
    """)
    
    st.divider()
    st.metric("Modelos Cargados", "3/3")

# ==========================================
# CARGA DE MODELOS
# ==========================================

@st.cache_resource
def load_classic_model():
    try:
        svm = joblib.load('svm_model.pkl')
        vocab = joblib.load('vocab_model.pkl')
        scaler = joblib.load('scaler.pkl')
        classes = joblib.load('class_names.pkl')
        return svm, vocab, scaler, classes, True
    except Exception as e:
        st.error(f"❌ Error cargando modelo clásico: {e}")
        return None, None, None, None, False

@st.cache_resource
def load_vit_model():
    model_path = "./final_model"
    try:
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(model_path)
        return model, processor, True
    except Exception as e:
        st.error(f"❌ Error cargando ViT: {e}")
        return None, None, False

@st.cache_resource
def load_convnext_model(num_classes=25):
    path = "best_ConvNeXt-Small_full.pth"
    
    # Verificar que el archivo existe
    if not os.path.exists(path):
        st.error(f"❌ Archivo no encontrado: {path}")
        return None, False
    
    try:
        # Crear modelo
        model = timm.create_model(
            "convnext_small.in12k_ft_in1k",
            pretrained=False,
            num_classes=num_classes
        )
        
        # Intentar múltiples métodos de carga
        checkpoint = None
        error_msgs = []
        
        # Método 1: torch.load con weights_only=False
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e1:
            error_msgs.append(f"Método 1: {str(e1)}")
            
            # Método 2: torch.load sin weights_only
            try:
                checkpoint = torch.load(path, map_location='cpu')
            except Exception as e2:
                error_msgs.append(f"Método 2: {str(e2)}")
                
                # Método 3: Con pickle_module
                try:
                    import pickle
                    checkpoint = torch.load(path, map_location='cpu', pickle_module=pickle)
                except Exception as e3:
                    error_msgs.append(f"Método 3: {str(e3)}")
        
        if checkpoint is None:
            raise Exception("No se pudo cargar el checkpoint con ningún método")
        
        # Determinar el formato del checkpoint
        if isinstance(checkpoint, dict):
            # Buscar las claves posibles
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Asumir que el dict ES el state_dict
                state_dict = checkpoint
            
            # Cargar state_dict
            model.load_state_dict(state_dict, strict=False)
        else:
            # Si checkpoint es directamente el modelo
            model = checkpoint
        
        model.eval()
        return model, True
        
    except Exception as e:
        st.error(f"❌ Error cargando ConvNeXt: {e}")
        
        # Información de debugging detallada
        with st.expander("🔍 Información de Debug"):
            st.code(f"Error principal: {str(e)}")
            
            if error_msgs:
                st.write("**Intentos de carga:**")
                for msg in error_msgs:
                    st.code(msg)
            
            # Información del archivo
            try:
                file_size = os.path.getsize(path) / (1024*1024)
                st.write(f"**Tamaño del archivo:** {file_size:.2f} MB")
                
                with open(path, 'rb') as f:
                    header = f.read(20)
                    st.write(f"**Primeros bytes:** {header.hex()}")
            except Exception as file_err:
                st.write(f"No se pudo leer info del archivo: {file_err}")
            
            st.write("**Posibles soluciones:**")
            st.write("1. El archivo puede estar corrupto - descárgalo de nuevo")
            st.write("2. Verifica que sea un archivo .pth válido de PyTorch")
            st.write("3. Si lo descargaste de Google Drive/Colab, asegúrate que no sea un HTML")
            st.write("4. Intenta exportar el modelo de nuevo desde tu notebook de entrenamiento")
        
        return None, False

# Cargar modelos
with st.spinner("🔄 Cargando modelos..."):
    svm, vocab, scaler, class_names, classic_ok = load_classic_model()
    vit_model, vit_processor, vit_ok = load_vit_model()
    
    num_classes = len(class_names) if class_names else 25
    convnext_model, convnext_ok = load_convnext_model(num_classes=num_classes)

# Estado de modelos
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    if classic_ok:
        st.success("✅ Modelo Clásico (SVM)")
    else:
        st.error("❌ SVM No Disponible")

with col_status2:
    if vit_ok:
        st.success("✅ Vision Transformer")
    else:
        st.error("❌ ViT No Disponible")

with col_status3:
    if convnext_ok:
        st.success("✅ ConvNeXt")
    else:
        st.error("❌ ConvNeXt No Disponible")

if not (classic_ok or vit_ok or convnext_ok):
    st.warning("""
    ⚠️ **No se pudieron cargar los modelos**
    
    Asegúrate de tener:
    1. `svm_model.pkl`, `vocab_model.pkl`, `scaler.pkl`, `class_names.pkl`
    2. Carpeta `./final_model` con el modelo ViT
    3. `best_ConvNeXt-Small_full.pth`
    """)
    st.stop()

# ==========================================
# FUNCIONES DE PREDICCIÓN
# ==========================================

def predict_classic(pil_image):
    """Predicción con modelo clásico ORB + SVM"""
    try:
        img_np = np.array(pil_image)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (224, 224))
        
        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(img_resized, None)
        
        if des is None:
            return "No detectado", 0.0, None, 0
        
        hist = np.zeros(vocab.n_clusters)
        predictions = vocab.predict(des.astype(float))
        for pred in predictions:
            hist[pred] += 1
        
        hist_scaled = scaler.transform([hist])
        probs = svm.predict_proba(hist_scaled)[0]
        idx = np.argmax(probs)
        
        # Top 3 predicciones
        top_indices = np.argsort(probs)[-3:][::-1]
        top_predictions = [(class_names[i], probs[i]) for i in top_indices]
        
        return class_names[idx], probs[idx], top_predictions, len(kp)
    
    except Exception as e:
        st.error(f"Error en predicción clásica: {e}")
        return "Error", 0.0, None, 0

def predict_vit(pil_image):
    """Predicción con Vision Transformer"""
    try:
        inputs = vit_processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = vit_model(**inputs)
        
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0]
        
        conf, idx = torch.max(probs, 0)
        label = vit_model.config.id2label[idx.item()]
        
        # Top 3 predicciones
        top_probs, top_indices = torch.topk(probs, 3)
        top_predictions = [
            (vit_model.config.id2label[i.item()], p.item()) 
            for i, p in zip(top_indices, top_probs)
        ]
        
        return label, conf.item(), top_predictions
    
    except Exception as e:
        st.error(f"Error en predicción ViT: {e}")
        return "Error", 0.0, None

def predict_convnext(pil_image):
    """Predicción con ConvNeXt"""
    try:
        val_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_tensor = val_tf(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            output = convnext_model(img_tensor)
            probs = F.softmax(output, dim=1)[0]
        
        conf, idx = torch.max(probs, 0)
        
        if class_names:
            label = class_names[idx.item()]
        else:
            label = f"Clase {idx.item()}"
        
        # Top 3 predicciones
        top_probs, top_indices = torch.topk(probs, 3)
        top_predictions = [
            (class_names[i.item()] if class_names else f"Clase {i.item()}", p.item())
            for i, p in zip(top_indices, top_probs)
        ]
        
        return label, conf.item(), top_predictions
    
    except Exception as e:
        st.error(f"Error en predicción ConvNeXt: {e}")
        return "Error", 0.0, None

def create_confidence_chart(predictions, title):
    """Crea gráfico de barras horizontal para top predicciones"""
    if not predictions:
        return None
    
    labels = [p[0] for p in predictions]
    values = [p[1] * 100 for p in predictions]
    
    colors = ['#667eea' if i == 0 else '#a8b3f5' for i in range(len(labels))]
    
    fig = go.Figure(data=[
        go.Bar(
            y=labels,
            x=values,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{v:.1f}%' for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Confianza (%)",
        yaxis_title="",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def get_confidence_color(conf):
    """Retorna color según nivel de confianza"""
    if conf > 0.8:
        return "🟢", "green"
    elif conf > 0.6:
        return "🟡", "orange"
    else:
        return "🔴", "red"

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

st.markdown("### 📸 Cargar Imagen")

uploaded_file = st.file_uploader(
    "Selecciona una imagen de arquitectura",
    type=["jpg", "png", "jpeg"],
    help="Formatos soportados: JPG, PNG, JPEG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Información de la imagen
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Resolución", f"{image.size[0]} x {image.size[1]}")
    with col_info2:
        st.metric("Formato", image.format or "RGB")
    with col_info3:
        st.metric("Modo", image.mode)
    
    # Mostrar imagen
    st.markdown("### 🖼️ Imagen Cargada")
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(image, caption="Vista previa", use_container_width=True)
    
    st.markdown("---")
    
    # Botón de análisis
    if st.button("🔍 Analizar con los 3 Modelos", use_container_width=True):
        
        start_time = datetime.now()
        
        # Barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown("### 📊 Resultados del Análisis")
        
        col_classic, col_vit, col_convnext = st.columns(3)
        
        # --- MODELO CLÁSICO ---
        with col_classic:
            st.markdown("#### 🧠 Modelo Clásico")
            st.caption("ORB + BoVW + SVM")
            
            if classic_ok:
                status_text.text("Procesando con modelo clásico...")
                progress_bar.progress(20)
                
                pred_c, conf_c, top_preds_c, num_keypoints = predict_classic(image)
                progress_bar.progress(33)
                
                emoji, color = get_confidence_color(conf_c)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {color};">{emoji} {pred_c}</h3>
                    <p><strong>Confianza:</strong> {conf_c:.1%}</p>
                    <p><strong>Puntos clave:</strong> {num_keypoints}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if top_preds_c:
                    with st.expander("📈 Ver top 3 predicciones"):
                        fig_classic = create_confidence_chart(top_preds_c, "Top 3 - SVM")
                        if fig_classic:
                            st.plotly_chart(fig_classic, use_container_width=True)
            else:
                st.error("❌ Modelo no disponible")

        # --- VISION TRANSFORMER ---
        with col_vit:
            st.markdown("#### 🤖 Vision Transformer")
            st.caption("Google ViT-Base")
            
            if vit_ok:
                status_text.text("Procesando con ViT...")
                progress_bar.progress(55)
                
                pred_v, conf_v, top_preds_v = predict_vit(image)
                progress_bar.progress(66)
                
                emoji, color = get_confidence_color(conf_v)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {color};">{emoji} {pred_v}</h3>
                    <p><strong>Confianza:</strong> {conf_v:.1%}</p>
                    <p><strong>Arquitectura:</strong> Transformer</p>
                </div>
                """, unsafe_allow_html=True)
                
                if top_preds_v:
                    with st.expander("📈 Ver top 3 predicciones"):
                        fig_vit = create_confidence_chart(top_preds_v, "Top 3 - ViT")
                        if fig_vit:
                            st.plotly_chart(fig_vit, use_container_width=True)
            else:
                st.error("❌ Modelo no disponible")

        # --- CONVNEXT ---
        with col_convnext:
            st.markdown("#### ⚡ ConvNeXt")
            st.caption("Meta ConvNeXt Small")
            
            if convnext_ok:
                status_text.text("Procesando con ConvNeXt...")
                progress_bar.progress(85)
                
                pred_cn, conf_cn, top_preds_cn = predict_convnext(image)
                progress_bar.progress(100)
                
                emoji, color = get_confidence_color(conf_cn)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {color};">{emoji} {pred_cn}</h3>
                    <p><strong>Confianza:</strong> {conf_cn:.1%}</p>
                    <p><strong>Arquitectura:</strong> CNN Moderna</p>
                </div>
                """, unsafe_allow_html=True)
                
                if top_preds_cn:
                    with st.expander("📈 Ver top 3 predicciones"):
                        fig_cn = create_confidence_chart(top_preds_cn, "Top 3 - ConvNeXt")
                        if fig_cn:
                            st.plotly_chart(fig_cn, use_container_width=True)
            else:
                st.error("❌ Modelo no disponible")
        
        # Limpiar barra de progreso
        progress_bar.empty()
        status_text.empty()
        
        # Tiempo de procesamiento
        elapsed_time = (datetime.now() - start_time).total_seconds()
        st.info(f"⏱️ Tiempo de procesamiento: {elapsed_time:.2f} segundos")
        
        # --- ANÁLISIS COMPARATIVO ---
        st.markdown("---")
        st.markdown("### 🔬 Análisis Comparativo")
        
        # Recopilar predicciones
        predictions = []
        if classic_ok:
            predictions.append(('SVM', pred_c, conf_c))
        if vit_ok:
            predictions.append(('ViT', pred_v, conf_v))
        if convnext_ok:
            predictions.append(('ConvNeXt', pred_cn, conf_cn))
        
        if len(predictions) >= 2:
            # Verificar consenso
            pred_labels = [p[1] for p in predictions]
            unique_preds = set(pred_labels)
            
            if len(unique_preds) == 1:
                # Todos coinciden
                st.success(f"""
                ### ✅ Consenso Total
                
                Los {len(predictions)} modelos coinciden en clasificar la arquitectura como **{pred_labels[0]}**.
                
                **Confianzas:**
                """)
                for name, pred, conf in predictions:
                    st.write(f"- **{name}**: {conf:.1%}")
                
                st.info("Este nivel de acuerdo indica una clasificación confiable.")
                
            else:
                # Hay diferencias
                st.warning(f"""
                ### ⚠️ Discrepancia Detectada
                
                Los modelos tienen diferentes interpretaciones:
                """)
                
                for name, pred, conf in predictions:
                    st.write(f"- **{name}**: {pred} ({conf:.1%})")
                
                # Encontrar el de mayor confianza
                max_conf_pred = max(predictions, key=lambda x: x[2])
                
                st.info(f"""
                **💡 Recomendación**: 
                
                Considera la predicción con mayor confianza: **{max_conf_pred[1]}** 
                del modelo {max_conf_pred[0]} (Confianza: {max_conf_pred[2]:.1%})
                
                Los modelos modernos (ViT y ConvNeXt) suelen ser más robustos ante:
                - Variaciones de iluminación
                - Diferentes ángulos de captura
                - Detalles arquitectónicos complejos
                """)
        
        # Botón para nuevo análisis
        st.markdown("---")
        if st.button("🔄 Analizar Nueva Imagen", use_container_width=True):
            st.rerun()

else:
    # Mensaje cuando no hay imagen
    st.info("""
    👆 **Sube una imagen para comenzar el análisis**
    
    - Formatos aceptados: JPG, PNG, JPEG
    - Recomendación: Imágenes claras con buena iluminación
    - Mejor resultado: Vista frontal del edificio
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Computer Vision • Deep Learning</p>
    <p><small>Versión 3.0 • 2024</small></p>
</div>
""", unsafe_allow_html=True)

#streamlit run app.py