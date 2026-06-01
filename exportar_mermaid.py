#!/usr/bin/env python3
"""
Script para exportar diagrama Mermaid a PNG usando mermaid-cli
"""
import subprocess
import os
from pathlib import Path

# Rutas
mmd_file = r"c:\usr\ths_mia_fiis\pln\trabajo\arquitectura_sistema.mmd"
output_png = r"c:\usr\ths_mia_fiis\pln\trabajo\arquitectura_sistema.png"

print("🔄 Exportando diagrama Mermaid...")

# Verificar si el archivo Mermaid existe
if not os.path.exists(mmd_file):
    print(f"❌ No encontrado: {mmd_file}")
    exit(1)

try:
    # Usar mermaid-cli (npx mmdc)
    print(f"📄 Archivo: {mmd_file}")
    print(f"💾 Salida: {output_png}\n")
    
    # Ejecutar mmdc
    cmd = [
        "npx", "-y", "mermaid-cli@latest",
        "-i", mmd_file,
        "-o", output_png,
        "-w", "1200",
        "-H", "800",
        "--theme", "default"
    ]
    
    print(f"⏳ Ejecutando: {' '.join(cmd[:5])}...\n")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode == 0:
        if os.path.exists(output_png):
            size = os.path.getsize(output_png) / 1024  # KB
            print(f"✅ Diagrama exportado exitosamente!")
            print(f"   Archivo: {output_png}")
            print(f"   Tamaño: {size:.1f} KB")
        else:
            print(f"⚠️ Comando ejecutado pero no encontré el archivo de salida")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
    else:
        print(f"❌ Error al ejecutar mermaid-cli")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
except FileNotFoundError:
    print("❌ npx no encontrado. Instalando mermaid-cli globalmente...")
    subprocess.run(["npm", "install", "-g", "mermaid-cli"], check=True)
    print("\n🔄 Reintentando exportación...\n")
    subprocess.run([
        "npx", "-y", "mermaid-cli@latest",
        "-i", mmd_file,
        "-o", output_png,
        "-w", "1200",
        "-H", "800"
    ], check=True)
    print(f"✅ Diagrama exportado exitosamente!")

except subprocess.TimeoutExpired:
    print("❌ Timeout: El proceso tardó demasiado")
except Exception as e:
    print(f"❌ Error: {e}")
