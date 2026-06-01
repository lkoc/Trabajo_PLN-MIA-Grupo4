#!/usr/bin/env python3
"""
Generar diagrama de arquitectura usando PIL
"""
from PIL import Image, ImageDraw, ImageFont
import os

# Crear imagen
width, height = 1400, 900
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# Intentar usar fuente disponible
try:
    font_small = ImageFont.truetype("arial.ttf", 9)
    font_med = ImageFont.truetype("arial.ttf", 11)
except:
    font_small = ImageFont.load_default()
    font_med = ImageFont.load_default()

# Colores
colors = {
    'blue': '#4A90E2',
    'green': '#50C878',
    'orange': '#FFB347',
    'red': '#FF6B6B',
    'purple': '#800080',
    'pink': '#FF1493',
    'lightgreen': '#32CD32',
    'cyan': '#00CED1',
    'yellow': '#FFD700',
    'light_pink': '#FFB6C1'
}

def hex_to_rgb(hex_color):
    if isinstance(hex_color, str):
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        else:
            return (0, 0, 0)  # Default si no es hex
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def draw_box(x, y, w, h, text, bg_color, text_color='white'):
    """Dibujar caja redondeada"""
    draw.rectangle([x, y, x+w, y+h], fill=hex_to_rgb(bg_color), outline='#333333', width=2)
    draw.text((x+w//2, y+h//2), text, fill=hex_to_rgb(text_color), font=font_med, anchor='mm')

def draw_arrow(x1, y1, x2, y2):
    """Dibujar flecha"""
    draw.line([x1, y1, x2, y2], fill='#333333', width=2)
    # Punta de flecha
    angle = 0.3
    draw.polygon([(x2, y2), (x2-10, y2-10), (x2-10, y2+10)], fill='#333333')

# Posiciones y cajas
boxes = [
    # Fila 1
    (50, 50, 150, 60, '🎬 Video\nYouTube', '#4A90E2'),
    (300, 50, 150, 60, '📥 Descarga\nAudio', '#50C878'),
    (550, 50, 150, 60, '🎙️ Transcripción\nWhisper', '#FFB347'),
    (800, 50, 150, 60, '✂️ Segmentación\nChunks', '#FF6B6B'),
    (1050, 50, 150, 60, '🏷️ Etiquetado\nHTML', '#800080'),
    
    # Fila 2
    (50, 200, 150, 60, '💾 Dataset\nSQLite', '#800080'),
    (300, 200, 150, 60, '🤖 Entrenamiento\nEnsemble', '#FF1493'),
    
    # Fila 3 - Modelos
    (200, 350, 100, 50, 'BERT', '#FFB6C1'),
    (320, 350, 100, 50, 'SVM', '#FFB6C1'),
    (440, 350, 100, 50, 'RoBERTa', '#FFB6C1'),
    
    # Fila 4
    (300, 500, 150, 60, '✅ Evaluación', '#32CD32'),
    
    # Fila 5
    (550, 650, 150, 60, '🚀 Producción\nLocal', '#00CED1'),
    (800, 650, 150, 60, '📊 Dashboard\nStreamlit', '#FFD700'),
]

# Dibujar cajas
for x, y, w, h, text, color in boxes:
    text_color = 'white' if color != '#FFD700' and color != '#FFB347' else 'black'
    draw_box(x, y, w, h, text, color, text_color)

# Dibujar flechas (conexiones)
arrows = [
    # Fila 1
    (200, 80, 300, 80),
    (450, 80, 550, 80),
    (700, 80, 800, 80),
    (950, 80, 1050, 80),
    
    # De fila 1 a fila 2
    (1050, 110, 125, 200),  # De etiquetado a dataset
    
    # De dataset a entrenamiento
    (200, 230, 300, 230),
    
    # De entrenamiento a modelos
    (320, 260, 250, 350),
    (350, 260, 350, 350),
    (380, 260, 480, 350),
    
    # De modelos a evaluación
    (250, 400, 350, 500),
    (350, 400, 375, 500),
    (480, 400, 400, 500),
    
    # De evaluación a producción
    (375, 560, 575, 650),
    
    # De producción a dashboard
    (700, 680, 800, 680),
]

for x1, y1, x2, y2 in arrows:
    draw_arrow(x1, y1, x2, y2)

# Título
try:
    title_font = ImageFont.truetype("arial.ttf", 16)
except:
    title_font = font_med
draw.text((700, 20), "Arquitectura del Sistema: Moderador de Contenido en Videos", 
          fill=hex_to_rgb('#000000'), font=title_font, anchor='mm')

# Guardar
output_path = r'c:\usr\ths_mia_fiis\pln\trabajo\arquitectura_sistema.png'
img.save(output_path)
print(f"✓ Diagrama generado: {output_path}")
