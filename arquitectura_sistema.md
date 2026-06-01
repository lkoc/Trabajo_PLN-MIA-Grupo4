graph TD
    A["🎬 Video YouTube/TikTok"] 
    B["📥 yt-dlp: Descarga Audio"]
    C["🎙️ Whisper ASR: Transcripción<br/>(Local, 16kHz)"]
    D["✂️ Segmentación en Chunks<br/>(30-60 seg)"]
    E["🏷️ Etiquetado Manual<br/>(HTML Framework)"]
    F["💾 Dataset SQLite<br/>(Incremental+Hash)"]
    G["🤖 Entrenamiento Ensemble"]
    G1["BERT Fine-tuned"]
    G2["SVM + TF-IDF"]
    G3["RoBERTa"]
    H["✅ Evaluación<br/>(70-15-15 split)"]
    I["🚀 Modelo en Producción"]
    J["📊 Dashboard Streamlit<br/>(Resultados)"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> G1
    G --> G2
    G --> G3
    G1 --> H
    G2 --> H
    G3 --> H
    H --> I
    I --> J
    
    style A fill:#4A90E2,color:#fff
    style B fill:#50C878,color:#fff
    style C fill:#FFB347,color:#000
    style D fill:#FF6B6B,color:#fff
    style E fill:#C71585,color:#fff
    style F fill:#800080,color:#fff
    style G fill:#FF1493,color:#fff
    style G1 fill:#FFB6C1
    style G2 fill:#FFB6C1
    style G3 fill:#FFB6C1
    style H fill:#32CD32,color:#fff
    style I fill:#00CED1,color:#fff
    style J fill:#FFD700,color:#000
