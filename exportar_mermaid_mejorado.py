#!/usr/bin/env python3
"""
Script para exportar Mermaid a PNG con calidad alta
Usa mermaid-cli con mejor manejo de dependencias
"""
import subprocess
import os
import sys
from pathlib import Path

def check_nodejs():
    """Verifica si Node.js está instalado"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Node.js encontrado: {result.stdout.strip()}")
            return True
    except:
        pass
    return False

def install_mermaid_cli():
    """Instala mermaid-cli globalmente"""
    try:
        print("📦 Instalando mermaid-cli...")
        subprocess.run(
            ["npm", "install", "-g", "@mermaid-js/mermaid-cli"],
            timeout=120,
            capture_output=True
        )
        print("✓ mermaid-cli instalado")
        return True
    except Exception as e:
        print(f"❌ Error instalando mermaid-cli: {e}")
        return False

def export_mermaid(input_file, output_file, width=1400, height=900):
    """Exporta Mermaid a PNG"""
    try:
        # Verificar si mmdc está disponible
        result = subprocess.run(
            ["mmdc", "-V"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print("⚠️ mermaid-cli no disponible, instalando...")
            if not install_mermaid_cli():
                print("❌ No se pudo instalar mermaid-cli")
                return False
        
        # Exportar con mmdc
        print(f"\n🎨 Exportando: {input_file}")
        print(f"   Dimensiones: {width}x{height}")
        
        cmd = [
            "mmdc",
            "-i", input_file,
            "-o", output_file,
            "-w", str(width),
            "-H", str(height),
            "-t", "default",
            "-b", "transparent"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_file):
            size_kb = os.path.getsize(output_file) / 1024
            print(f"✅ Exportado exitosamente!")
            print(f"   Archivo: {output_file}")
            print(f"   Tamaño: {size_kb:.1f} KB")
            return True
        else:
            print(f"❌ Error exportando:")
            print(f"   Stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Rutas
    mmd_file = r"c:\usr\ths_mia_fiis\pln\trabajo\arquitectura_sistema.mmd"
    png_file = r"c:\usr\ths_mia_fiis\pln\trabajo\arquitectura_sistema.png"
    
    print("=" * 60)
    print("🎨 EXPORTADOR MERMAID → PNG")
    print("=" * 60)
    
    # Verificar que el archivo .mmd existe
    if not os.path.exists(mmd_file):
        print(f"❌ No encontrado: {mmd_file}")
        sys.exit(1)
    
    print(f"📄 Archivo: {mmd_file}\n")
    
    # Verificar Node.js
    if not check_nodejs():
        print("\n⚠️ Node.js no encontrado")
        print("   Por favor instala Node.js desde: https://nodejs.org/")
        sys.exit(1)
    
    # Exportar con dimensiones optimizadas para slide (75% del ancho)
    # Slide 16:9 típicamente 945x532, 75% ≈ 710x532
    # Pero aumentamos para mejor calidad: 1400x900 (después se escala en LaTeX)
    success = export_mermaid(mmd_file, png_file, width=1400, height=900)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Proceso completado exitosamente!")
        print("=" * 60)
        print("\n🔍 Próximos pasos:")
        print("1. Abre presentacion_grupo4.tex")
        print("2. El slide 3 mostrará el diagrama actualizado")
        print("3. Compila: pdflatex -interaction=nonstopmode presentacion_grupo4.tex")
    else:
        print("\n" + "=" * 60)
        print("❌ Exportación fallida")
        print("=" * 60)
        sys.exit(1)
