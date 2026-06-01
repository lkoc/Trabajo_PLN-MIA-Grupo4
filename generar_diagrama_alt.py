#!/usr/bin/env python3
"""
Script para generar diagrama de arquitectura usando graphviz/matplotlib
"""
import os
try:
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
except ImportError:
    print("Instalando Pillow...")
    import subprocess
    subprocess.run(["pip", "install", "Pillow"], check=True)
    from PIL import Image, ImageDraw, ImageFont

def create_architecture_diagram():
    """Crea un diagrama de arquitectura del sistema"""
    
    # Dimensiones
    width, height = 1200, 800
    bg_color = (255, 255, 255)
    line_color = (100, 100, 100)
    
    # Crear imagen
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Intentar cargar fuente, si no usar default
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 12)
        small_font = ImageFont.truetype("arial.ttf", 10)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Colores por etapa
    colors = {
        'input': (225, 245, 255),      # azul muy claro
        'download': (179, 229, 252),   # azul claro
        'process': (129, 212, 250),    # azul
        'transcribe': (79, 195, 247),  # azul oscuro
        'segment': (41, 182, 246),     # azul más oscuro
        'label': (3, 169, 244),        # azul
        'storage': (3, 155, 229),      # azul oscuro
        'train': (2, 136, 209),        # azul muy oscuro
        'model': (2, 119, 189),        # azul
        'inference': (1, 87, 155),     # azul muy oscuro
        'output': (21, 101, 192)       # azul profundo
    }
    
    # Dibujar título
    draw.text((30, 20), "Arquitectura General del Sistema - Moderador de Contenido", 
              fill=(0, 0, 0), font=title_font)
    
    # Posiciones y cajas (x, y, ancho, alto)
    boxes = [
        # Row 1
        (50, 80, "🎬 Video Input\n(URL YouTube)", colors['input']),
        (250, 80, "🔊 Audio Download\n(MP4/WebM)", colors['download']),
        (450, 80, "📊 Audio Processing\n(16kHz, Mono, WAV)", colors['process']),
        (650, 80, "📝 Transcription\n(Spanish Text)", colors['transcribe']),
        (850, 80, "✂️ Segmentation\n(30-60s chunks)", colors['segment']),
        
        # Row 2
        (50, 250, "🏷️ Labeled Dataset\n(7 categories)", colors['label']),
        (250, 250, "🗄️ SQLite DB\n(MD5 Dedup)", colors['storage']),
        (450, 250, "🤖 Model Training\n(BERT/RoBERTa)", colors['train']),
        (650, 250, "✅ Production Model\n(ONNX Format)", colors['model']),
        (850, 250, "🎯 Content Moderation\n(Local, No API)", colors['inference']),
        
        # Row 3
        (300, 420, "📊 Reports & Monitoring\n📈 Analytics Dashboard", colors['output']),
    ]
    
    # Dibujar cajas
    box_positions = {}
    for i, (x, y, text, color) in enumerate(boxes):
        box_width, box_height = 150, 120
        # Dibujar rectángulo
        draw.rectangle([x, y, x+box_width, y+box_height], fill=color, outline=(0, 0, 0), width=2)
        
        # Dibujar texto
        lines = text.split('\n')
        text_y = y + 15
        for line in lines:
            draw.text((x + 10, text_y), line, fill=(0, 0, 0), font=small_font)
            text_y += 30
        
        # Guardar posición para flechas
        box_positions[i] = (x + box_width/2, y + box_height/2, box_width, box_height)
    
    # Dibujar flechas
    arrows = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Row 1
        (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),  # Row 2
        (9, 10),                              # to output
    ]
    
    for start, end in arrows:
        if start < len(box_positions) and end < len(box_positions):
            x1, y1, w1, h1 = box_positions[start]
            x2, y2, w2, h2 = box_positions[end]
            
            # Calcular puntos de conexión
            start_x = x1 + w1/2
            start_y = y1 + h1/2
            end_x = x2 - w2/2 if x2 > x1 else x2 + w2/2
            end_y = y2 - h2/2 if y2 > y1 else y2 + h2/2
            
            # Dibujar línea
            draw.line([start_x, start_y, end_x, end_y], fill=line_color, width=2)
            
            # Dibujar punta de flecha
            arrow_size = 10
            angle = 0.4
            end_x_adj = end_x - arrow_size * (1 if end_x >= start_x else -1)
            draw.line([end_x, end_y, end_x_adj, end_y - arrow_size], fill=line_color, width=2)
            draw.line([end_x, end_y, end_x_adj, end_y + arrow_size], fill=line_color, width=2)
    
    # Dibujar leyenda/descripción
    desc_y = 580
    description = """
    PIPELINE COMPLETO:
    1. Descarga de videos desde YouTube (canal específicos peruanos)
    2. Procesamiento de audio a 16kHz mono WAV
    3. Transcripción automática con Whisper (modelo local)
    4. Segmentación en chunks de 30-60 segundos
    5. Etiquetado manual en 7 categorías de moderación
    6. Limpieza de datos (deduplicación con MD5)
    7. Entrenamiento de modelo BERT/RoBERTa
    8. Exportación a formato ONNX para producción
    9. Inferencia 100% local sin dependencias externas
    10. Generación de reportes y análisis
    """
    
    for line in description.strip().split('\n'):
        draw.text((50, desc_y), line, fill=(50, 50, 50), font=small_font)
        desc_y += 18
    
    # Guardar
    output_path = r"c:\usr\ths_mia_fiis\pln\trabajo\arquitectura_sistema.png"
    img.save(output_path)
    print(f"✅ Diagrama creado: {output_path}")
    return output_path

if __name__ == "__main__":
    create_architecture_diagram()
