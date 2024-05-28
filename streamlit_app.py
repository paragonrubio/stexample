import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import numpy as np
import pandas as pd

# Configurar la página de Streamlit
st.title("Clasificación de imágenes con ResNet 2-")

st.dataframe(pd.read_csv("imdb-movies-dataset.csv"))


# Cargar el modelo ResNet50 preentrenado
model = models.resnet50(pretrained=True)
model.eval()

# Definir las transformaciones para la imagen
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar etiquetas de ImageNet
@st.cache_data
def load_labels():
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        response.raise_for_status()
        labels = response.text.splitlines()
        return labels
    except requests.exceptions.RequestException as e:
        st.error(f"Error al cargar las etiquetas: {e}")
        return []

labels = load_labels()

# Subir la imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Abrir la imagen
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen subida", use_column_width=True)
        st.write("")
        st.write("Clasificando...")

        # Preprocesar la imagen
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # Realizar la inferencia
        with torch.no_grad():
            out = model(batch_t)

        # Obtener la predicción
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        label = labels[index[0]] if labels else "Etiqueta no disponible"

        # Mostrar los resultados
        st.write(f"Predicción: **{label}**")
        st.write(f"Confianza: **{percentage[index[0]].item():.2f}%**")
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
