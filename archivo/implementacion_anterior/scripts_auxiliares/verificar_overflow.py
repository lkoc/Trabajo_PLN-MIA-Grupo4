#!/usr/bin/env python3
"""
Script para convertir páginas del PDF a imágenes y verificar overflow
"""
import os
import subprocess
from pathlib import Path

# Verificar si pdf2image está instalado
try:
    from pdf2image import convert_from_path
    print("✓ pdf2image está instalado")
except ImportError:
    print("Instalando pdf2image y poppler...")
    subprocess.run(["pip", "install", "pdf2image", "pillow"], check=True)
    from pdf2image import convert_from_path

# Ruta del PDF
pdf_path = r"c:\usr\ths_mia_fiis\pln\trabajo\presentacion_grupo4.pdf"
output_dir = r"c:\usr\ths_mia_fiis\pln\trabajo\preview_pages"

# Crear directorio de salida
os.makedirs(output_dir, exist_ok=True)

print(f"\n📄 Convirtiendo PDF: {pdf_path}")
print(f"📁 Guardando imágenes en: {output_dir}\n")

try:
    # Convertir todas las páginas a imágenes
    print("⏳ Procesando PDF (esto puede tomar un minuto)...")
    images = convert_from_path(pdf_path, dpi=150)
    
    print(f"✓ PDF tiene {len(images)} páginas\n")
    
    # Guardar cada página como imagen
    for i, image in enumerate(images, 1):
        output_file = os.path.join(output_dir, f"page_{i:02d}.png")
        image.save(output_file, "PNG")
        
        # Mostrar información de la página
        width, height = image.size
        aspect_ratio = width / height
        print(f"📄 Página {i:2d}: {width}x{height} px | Aspect ratio: {aspect_ratio:.2f}")
    
    print(f"\n✅ Conversión completada!")
    print(f"📁 Imágenes guardadas en: {output_dir}")
    print(f"\nAbre la carpeta 'preview_pages' en VS Code para revisar cada página")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nIntentando instalar dependencias necesarias...")
    subprocess.run(["pip", "install", "--upgrade", "pdf2image", "pillow"], check=True)
