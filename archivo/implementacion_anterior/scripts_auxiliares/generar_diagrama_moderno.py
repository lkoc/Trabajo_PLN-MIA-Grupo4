#!/usr/bin/env python3
"""
Generador de diagrama de arquitectura con diseño moderno
Usa Pillow (PIL) para crear una imagen visualmente atractiva
"""
from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_modern_architecture_diagram():
    """Crea un diagrama moderno con degradados y estilos profesionales"""
    
    # Dimensiones
    width, height = 1400, 900
    bg_color = (248, 249, 250)  # Gris muy claro
    
    # Crear imagen
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Cargar fuentes
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        stage_font = ImageFont.truetype("arial.ttf", 14)
        label_font = ImageFont.truetype("arial.ttf", 11)
        step_font = ImageFont.truetype("arial.ttf", 10)
    except:
        title_font = ImageFont.load_default()
        stage_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        step_font = ImageFont.load_default()
    
    # Paleta de colores moderna (degradado azul)
    colors = [
        {'fill': (227, 242, 253), 'border': (21, 101, 192), 'text': (13, 71, 161)},      # 0: Azul muy claro
        {'fill': (187, 222, 251), 'border': (25, 118, 210), 'text': (13, 71, 161)},      # 1
        {'fill': (144, 202, 249), 'border': (30, 136, 229), 'text': (255, 255, 255)},    # 2
        {'fill': (100, 181, 246), 'border': (33, 150, 243), 'text': (255, 255, 255)},    # 3
        {'fill': (66, 165, 245), 'border': (25, 103, 210), 'text': (255, 255, 255)},     # 4
        {'fill': (33, 150, 243), 'border': (21, 101, 192), 'text': (255, 255, 255)},     # 5: Azul
        {'fill': (30, 136, 229), 'border': (13, 71, 161), 'text': (255, 255, 255)},      # 6
        {'fill': (21, 101, 192), 'border': (13, 71, 161), 'text': (255, 255, 255)},      # 7
        {'fill': (13, 71, 161), 'border': (10, 53, 147), 'text': (255, 255, 255)},       # 8: Azul oscuro
    ]
    
    # Definir etapas del pipeline
    stages = [
        # Etapa 1: Entrada (fila superior)
        {'x': 50, 'y': 60, 'w': 130, 'h': 100, 'emoji': '🎬', 'title': 'INPUT', 'desc': 'Video URLs\nYouTube', 'step': '1', 'color': 0},
        
        # Etapa 2-5: Procesamiento (fila superior)
        {'x': 200, 'y': 60, 'w': 130, 'h': 100, 'emoji': '🔊', 'title': 'DESCARGA', 'desc': 'Audio MP4/WebM\nyt-dlp', 'step': '2', 'color': 1},
        {'x': 350, 'y': 60, 'w': 130, 'h': 100, 'emoji': '📊', 'title': 'PROCESAMIENTO', 'desc': '16kHz, Mono\nWAV', 'step': '3', 'color': 2},
        {'x': 500, 'y': 60, 'w': 130, 'h': 100, 'emoji': '📝', 'title': 'TRANSCRIPCIÓN', 'desc': 'Whisper ASR\nSpanish', 'step': '4', 'color': 3},
        {'x': 650, 'y': 60, 'w': 130, 'h': 100, 'emoji': '✂️', 'title': 'SEGMENTACIÓN', 'desc': '30-60s chunks\nText', 'step': '5', 'color': 4},
        
        # Etapa 6-7: Etiquetado y almacenamiento (fila media)
        {'x': 800, 'y': 60, 'w': 130, 'h': 100, 'emoji': '🏷️', 'title': 'ETIQUETADO', 'desc': '7 Categorías\nManual', 'step': '6', 'color': 5},
        {'x': 950, 'y': 60, 'w': 130, 'h': 100, 'emoji': '🗄️', 'title': 'ALMACENAMIENTO', 'desc': 'SQLite DB\nMD5 Dedup', 'step': '7', 'color': 6},
        
        # Etapa 8-9: Modelo (fila media)
        {'x': 1100, 'y': 60, 'w': 130, 'h': 100, 'emoji': '🤖', 'title': 'ENTRENAMIENTO', 'desc': 'BERT/RoBERTa\nFine-tuning', 'step': '8', 'color': 7},
        {'x': 1250, 'y': 60, 'w': 130, 'h': 100, 'emoji': '✅', 'title': 'MODELO', 'desc': 'ONNX Format\nProduction', 'step': '9', 'color': 8},
        
        # Etapa 10-11: Salida (fila inferior)
        {'x': 125, 'y': 230, 'w': 130, 'h': 100, 'emoji': '🎯', 'title': 'INFERENCIA', 'desc': 'Local 100%\nNo APIs', 'step': '10', 'color': 7},
        {'x': 275, 'y': 230, 'w': 130, 'h': 100, 'emoji': '📊', 'title': 'REPORTES', 'desc': 'Analytics\nMonitoring', 'step': '11', 'color': 8},
    ]
    
    # Dibujar título
    title = "ARQUITECTURA GENERAL - MODERADOR DE CONTENIDO"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 380), title, fill=(13, 71, 161), font=title_font)
    
    # Dibujar etapas (cajas)
    stage_positions = {}
    for idx, stage in enumerate(stages):
        x, y, w, h = stage['x'], stage['y'], stage['w'], stage['h']
        color_idx = stage['color']
        c = colors[color_idx]
        
        # Dibujar caja redondeada (aproximado)
        draw.rectangle([x, y, x+w, y+h], fill=c['fill'], outline=c['border'], width=3)
        
        # Número de paso en la esquina superior derecha
        step_text = stage['step']
        draw.ellipse([x+w-25, y-10, x+w+5, y+15], fill=c['border'], outline=c['border'])
        draw.text((x+w-22, y-7), step_text, fill=(255, 255, 255), font=label_font)
        
        # Emoji (grande)
        draw.text((x+w//2-20, y+10), stage['emoji'], font=stage_font)
        
        # Título
        draw.text((x+10, y+30), stage['title'], fill=c['text'], font=stage_font)
        
        # Descripción
        desc_y = y + 50
        for line in stage['desc'].split('\n'):
            draw.text((x+10, desc_y), line, fill=c['text'], font=step_font)
            desc_y += 18
        
        # Guardar posición para flechas
        stage_positions[idx] = (x + w/2, y + h/2, w, h)
    
    # Dibujar flechas conectoras
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 9), (9, 10),
    ]
    
    arrow_color = (21, 101, 192)
    for start_idx, end_idx in connections:
        if start_idx in stage_positions and end_idx in stage_positions:
            x1, y1, w1, h1 = stage_positions[start_idx]
            x2, y2, w2, h2 = stage_positions[end_idx]
            
            # Punto de salida (lado derecho)
            start_x = x1 + w1/2 + 5
            start_y = y1
            
            # Punto de entrada (lado izquierdo)
            end_x = x2 - w2/2 - 5
            end_y = y2
            
            # Dibujar línea con curvatura
            # Para simplificar, usamos línea recta
            draw.line([start_x, start_y, end_x, end_y], fill=arrow_color, width=3)
            
            # Punta de flecha
            import math
            angle = math.atan2(end_y - start_y, end_x - start_x)
            arrow_size = 12
            
            # Punto1 (lado superior)
            p1_x = end_x - arrow_size * math.cos(angle - math.pi/6)
            p1_y = end_y - arrow_size * math.sin(angle - math.pi/6)
            
            # Punto2 (lado inferior)
            p2_x = end_x - arrow_size * math.cos(angle + math.pi/6)
            p2_y = end_y - arrow_size * math.sin(angle + math.pi/6)
            
            # Dibujar triángulo
            draw.polygon([
                (end_x, end_y),
                (p1_x, p1_y),
                (p2_x, p2_y)
            ], fill=arrow_color, outline=arrow_color)
    
    # Dibujar leyenda/descripción en la parte inferior
    legend_y = 480
    legend_text = [
        "PIPELINE COMPLETO: Video Input → Descarga Audio → Procesamiento → Transcripción → Segmentación →",
        "Etiquetado Manual → Almacenamiento (SQLite) → Entrenamiento (BERT/RoBERTa) → Exportación (ONNX) →",
        "Inferencia Local 100% (Sin dependencias externas) → Generación de Reportes y Analytics"
    ]
    
    draw.text((50, legend_y), "FLUJO DEL SISTEMA:", fill=(13, 71, 161), font=label_font)
    legend_y += 25
    
    for line in legend_text:
        draw.text((50, legend_y), line, fill=(66, 66, 66), font=step_font)
        legend_y += 20
    
    # Guardar
    output_path = r"c:\usr\ths_mia_fiis\pln\trabajo\arquitectura_sistema.png"
    img.save(output_path, 'PNG', quality=95)
    
    print(f"✅ Diagrama creado exitosamente!")
    print(f"   Archivo: {output_path}")
    print(f"   Dimensión: {width}x{height}")
    print(f"   Tamaño: {len(img.tobytes()) / 1024 / 1024:.2f} MB (sin comprimir)")
    
    return output_path

if __name__ == "__main__":
    print("🎨 Generando diagrama moderno de arquitectura...")
    print()
    create_modern_architecture_diagram()
