"""Bibliografía IEEE verificada y utilidades de citado para cuadernos activos.

Las claves coinciden con ``Documento_final_paper/referencias.bib``. Las entradas
nuevas de documentación oficial se mantienen también en ese archivo maestro.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


CITATION_MARKER = re.compile(r"\[@([A-Za-z0-9_]+)\]")


REFERENCE_LIBRARY = {
    "ytdlp2026": (
        'yt-dlp contributors, "yt-dlp: A Feature-Rich Command-Line Audio/Video Downloader," '
        "GitHub repository, 2026. [Online]. Available: https://github.com/yt-dlp/yt-dlp. "
        "Accessed: Aug. 5, 2026."
    ),
    "depoix2026transcript": (
        'J. Depoix and contributors, "YouTube Transcript API: Python API for Retrieving '
        'YouTube Transcripts and Subtitles," GitHub repository, 2026. [Online]. Available: '
        "https://github.com/jdepoix/youtube-transcript-api. Accessed: Aug. 6, 2026."
    ),
    "youtube2023terms": (
        'YouTube, "Terms of Service," Nov. 2023. [Online]. Available: '
        "https://www.youtube.com/t/terms. Accessed: Aug. 5, 2026."
    ),
    "aoir2020ethics": (
        'A. S. franzke, A. Bechmann, M. Zimmer, et al., "Internet Research: Ethical '
        'Guidelines 3.0," Association of Internet Researchers, 2020. [Online]. Available: '
        "https://aoir.org/reports/ethics3.pdf"
    ),
    "tatman2017captions": (
        'R. Tatman, "Gender and Dialect Bias in YouTube\'s Automatic Captions," in '
        "Proc. 1st ACL Workshop Ethics NLP, 2017, pp. 53–59, doi: 10.18653/v1/W17-1606."
    ),
    "unicode2025normalization": (
        'Unicode Consortium, "Unicode Normalization Forms," Unicode Standard Annex '
        "No. 15, rev. 57, Jul. 2025. [Online]. Available: "
        "https://www.unicode.org/reports/tr15/tr15-57.html"
    ),
    "nist2015sha": (
        'National Institute of Standards and Technology, "Secure Hash Standard (SHS)," '
        "FIPS PUB 180-4, Aug. 2015, doi: 10.6028/NIST.FIPS.180-4."
    ),
    "fairstein2024balancing": (
        'Y. Fairstein, O. Kalinsky, Z. Karnin, et al., "Class Balancing for Efficient '
        'Active Learning in Imbalanced Datasets," in Proc. 18th Linguistic Annotation '
        "Workshop, 2024, pp. 77–86, doi: 10.18653/v1/2024.law-1.8."
    ),
    "huang2021balancing": (
        'Y. Huang, B. Giledereli, A. Köksal, et al., "Balancing Methods for Multi-label '
        'Text Classification with Long-Tailed Class Distribution," in Proc. EMNLP, 2021, '
        "pp. 8153–8161, doi: 10.18653/v1/2021.emnlp-main.643."
    ),
    "ollama2026structured": (
        'Ollama, "Structured Outputs," Ollama Documentation, 2026. [Online]. Available: '
        "https://docs.ollama.com/capabilities/structured-outputs. Accessed: Aug. 5, 2026."
    ),
    "ollama2026qwen35": (
        'Ollama, "Model Card: qwen3.5:4b," Ollama Model Library, 2026, model digest '
        "2a654d98e6fb. [Online]. Available: https://ollama.com/library/qwen3.5:4b. "
        "Accessed: Aug. 5, 2026."
    ),
    "ollama2026gemma34b": (
        'Ollama, "Model Card: gemma3:4b," Ollama Model Library, 2026. [Online]. '
        "Available: https://ollama.com/library/gemma3:4b. Accessed: Aug. 6, 2026."
    ),
    "qwen2025qwen3": (
        'A. Yang, A. Li, B. Yang, et al., "Qwen3 Technical Report," arXiv:2505.09388, '
        "2025, doi: 10.48550/arXiv.2505.09388."
    ),
    "hf2026qwen4bcard": (
        'Qwen Team, "Model Card: Qwen/Qwen3-4B," Hugging Face Hub, revision '
        "1cfa9a7208912126459214e8b04321603b3df60c, 2025. [Online]. Available: "
        "https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c. "
        "Accessed: Aug. 5, 2026."
    ),
    "hf2026qwen17bcard": (
        'Qwen Team, "Model Card: Qwen/Qwen3-1.7B," Hugging Face Hub, revision '
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, 2025. [Online]. Available: "
        "https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e. "
        "Accessed: Aug. 7, 2026."
    ),
    "googlecolab2026vscode": (
        'Google Colab, "Known Issues and Workarounds," googlecolab/colab-vscode Wiki, '
        "2026. [Online]. Available: https://github.com/googlecolab/colab-vscode/wiki/"
        "Known-Issues-and-Workarounds. Accessed: Aug. 5, 2026."
    ),
    "googlecolab2026faq": (
        'Google Colab, "Frequently Asked Questions," Google Research, 2026. [Online]. '
        "Available: https://research.google.com/colaboratory/faq.html. "
        "Accessed: Aug. 7, 2026."
    ),
    "schroeder2025llmassisted": (
        'H. Schroeder, D. Roy, and J. Kabbara, "Just Put a Human in the Loop? '
        'Investigating LLM-Assisted Annotation for Subjective Tasks," in Findings ACL, '
        "2025, pp. 25771–25795, doi: 10.18653/v1/2025.findings-acl.1323."
    ),
    "deepseek2026v4": (
        'DeepSeek, "DeepSeek V4 Preview Release," DeepSeek API Documentation, Apr. 2026. '
        "[Online]. Available: https://api-docs.deepseek.com/news/news260424/. "
        "Accessed: Aug. 5, 2026."
    ),
    "deepseek2026pricing": (
        'DeepSeek, "Models and Pricing," DeepSeek API Documentation, 2026. '
        "[Online]. Available: https://api-docs.deepseek.com/quick_start/pricing. "
        "Accessed: Aug. 7, 2026."
    ),
    "settles2009active": (
        'B. Settles, "Active Learning Literature Survey," Univ. Wisconsin–Madison, '
        "Computer Sciences Tech. Rep. 1648, 2009. [Online]. Available: "
        "https://minds.wisconsin.edu/handle/1793/60660"
    ),
    "bourgeade2024context": (
        'T. Bourgeade, Z. Li, F. Benamara, et al., "Humans Need Context, What about '
        'Machines? Investigating Conversational Context in Abusive Language Detection," '
        "in Proc. LREC-COLING, 2024, pp. 8438–8452. [Online]. Available: "
        "https://aclanthology.org/2024.lrec-main.740/"
    ),
    "artstein2008agreement": (
        'R. Artstein and M. Poesio, "Inter-Coder Agreement for Computational Linguistics," '
        "Comput. Linguistics, vol. 34, no. 4, pp. 555–596, 2008, "
        "doi: 10.1162/coli.07-034-R2."
    ),
    "choi2024llmeffect": (
        'A. S. Choi, S. S. Akter, J. P. Singh, et al., "The LLM Effect: Are Humans Truly '
        'Using LLMs, or Are They Being Influenced By Them Instead?" in Proc. EMNLP, 2024, '
        "pp. 22032–22054, doi: 10.18653/v1/2024.emnlp-main.1230."
    ),
    "salton1988tfidf": (
        'G. Salton and C. Buckley, "Term-Weighting Approaches in Automatic Text Retrieval," '
        "Inf. Process. Manage., vol. 24, no. 5, pp. 513–523, 1988, "
        "doi: 10.1016/0306-4573(88)90021-0."
    ),
    "cox1958logistic": (
        'D. R. Cox, "The Regression Analysis of Binary Sequences," J. Roy. Stat. Soc. B, '
        "vol. 20, no. 2, pp. 215–232, 1958, doi: 10.1111/j.2517-6161.1958.tb00292.x."
    ),
    "cortes1995svm": (
        'C. Cortes and V. Vapnik, "Support-Vector Networks," Mach. Learn., vol. 20, '
        "pp. 273–297, 1995, doi: 10.1007/BF00994018."
    ),
    "rennie2003cnb": (
        'J. D. M. Rennie, L. Shih, J. Teevan, et al., "Tackling the Poor Assumptions of '
        'Naive Bayes Text Classifiers," in Proc. ICML, 2003, pp. 616–623. [Online]. '
        "Available: https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf"
    ),
    "bottou2010sgd": (
        'L. Bottou, "Large-Scale Machine Learning with Stochastic Gradient Descent," in '
        "Proc. COMPSTAT, 2010, pp. 177–186, doi: 10.1007/978-3-7908-2604-3_16."
    ),
    "pedregosa2011sklearn": (
        'F. Pedregosa, G. Varoquaux, A. Gramfort, et al., "Scikit-Learn: Machine Learning '
        'in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011. [Online]. '
        "Available: https://www.jmlr.org/papers/v12/pedregosa11a.html"
    ),
    "tsoumakas2007multilabel": (
        'G. Tsoumakas and I. Katakis, "Multi-Label Classification: An Overview," Int. J. '
        "Data Warehousing Mining, vol. 3, no. 3, pp. 1–13, 2007, "
        "doi: 10.4018/jdwm.2007070101."
    ),
    "vaswani2017attention": (
        'A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention Is All You Need," in Adv. '
        "Neural Inf. Process. Syst., vol. 30, pp. 5998–6008, 2017."
    ),
    "wang2020minilm": (
        'W. Wang, F. Wei, L. Dong, et al., "MiniLM: Deep Self-Attention Distillation for '
        'Task-Agnostic Compression of Pre-Trained Transformers," in Adv. Neural Inf. '
        "Process. Syst., vol. 33, 2020. [Online]. Available: "
        "https://proceedings.neurips.cc/paper/2020/hash/"
        "3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"
    ),
    "reimers2020multilingual": (
        'N. Reimers and I. Gurevych, "Making Monolingual Sentence Embeddings Multilingual '
        'Using Knowledge Distillation," in Proc. EMNLP, 2020, pp. 4512–4525, '
        "doi: 10.18653/v1/2020.emnlp-main.365."
    ),
    "wang2024e5": (
        'L. Wang, N. Yang, X. Huang, et al., "Multilingual E5 Text Embeddings: A Technical '
        'Report," arXiv:2402.05672, 2024, doi: 10.48550/arXiv.2402.05672.'
    ),
    "hf2026minilmcard": (
        'Sentence Transformers, "Model Card: sentence-transformers/paraphrase-multilingual-'
        'MiniLM-L12-v2," Hugging Face Hub, revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42, '
        "2026. [Online]. Available: https://huggingface.co/sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    ),
    "hf2026e5card": (
        'intfloat, "Model Card: intfloat/multilingual-e5-small," Hugging Face Hub, '
        "revision 614241f622f53c4eeff9890bdc4f31cfecc418b3, 2026. [Online]. Available: "
        "https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3"
    ),
    "wolf2020transformers": (
        'T. Wolf, L. Debut, V. Sanh, et al., "Transformers: State-of-the-Art Natural '
        'Language Processing," in Proc. EMNLP: System Demonstrations, 2020, pp. 38–45, '
        "doi: 10.18653/v1/2020.emnlp-demos.6."
    ),
    "silla2011hierarchical": (
        'C. N. Silla Jr. and A. A. Freitas, "A Survey of Hierarchical Classification Across '
        'Different Application Domains," Data Min. Knowl. Discovery, vol. 22, no. 1–2, '
        "pp. 31–72, 2011, doi: 10.1007/s10618-010-0175-9."
    ),
    "zhou2020hiagm": (
        'J. Zhou, C. Ma, D. Long, et al., "Hierarchy-Aware Global Model for Hierarchical '
        'Text Classification," in Proc. ACL, 2020, pp. 1106–1117, '
        "doi: 10.18653/v1/2020.acl-main.104."
    ),
    "caruana1997multitask": (
        'R. Caruana, "Multitask Learning," Mach. Learn., vol. 28, pp. 41–75, 1997, '
        "doi: 10.1023/A:1007379606734."
    ),
    "hu2022lora": (
        'E. J. Hu, Y. Shen, P. Wallis, et al., "LoRA: Low-Rank Adaptation of Large Language '
        'Models," in Proc. ICLR, 2022. [Online]. Available: '
        "https://openreview.net/forum?id=nZeVKeeFYf9"
    ),
    "hf2026qwen06bcard": (
        'Qwen Team, "Model Card: Qwen/Qwen3-0.6B-Base," Hugging Face Hub, revision '
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd, 2025. [Online]. Available: "
        "https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    ),
    "hf2026peft018": (
        'Hugging Face, "LoRA API Reference," PEFT documentation, ver. 0.18.0, 2025. '
        "[Online]. Available: https://huggingface.co/docs/peft/v0.18.0/package_reference/lora. "
        "Accessed: Aug. 5, 2026."
    ),
    "saito2015pr": (
        'T. Saito and M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than '
        'the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," PLOS ONE, '
        "vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432."
    ),
    "efron1979bootstrap": (
        'B. Efron, "Bootstrap Methods: Another Look at the Jackknife," The Annals '
        "of Statistics, vol. 7, no. 1, pp. 1–26, 1979, "
        "doi: 10.1214/aos/1176344552."
    ),
    "field2007clusterbootstrap": (
        'C. A. Field and A. H. Welsh, "Bootstrapping Clustered Data," Journal of '
        "the Royal Statistical Society: Series B, vol. 69, no. 3, pp. 369–390, "
        "2007, doi: 10.1111/j.1467-9868.2007.00593.x."
    ),
    "blackwelder1982null": (
        'W. C. Blackwelder, "Proving the Null Hypothesis in Clinical Trials," '
        "Controlled Clinical Trials, vol. 3, no. 4, pp. 345–353, 1982, "
        "doi: 10.1016/0197-2456(82)90024-1."
    ),
    "dror2018significance": (
        "R. Dror, G. Baumer, S. Shlomov, et al., \"The Hitchhiker's Guide to Testing "
        'Statistical Significance in Natural Language Processing," in Proc. 56th '
        "Annual Meeting ACL, 2018, pp. 1383–1392, "
        "doi: 10.18653/v1/P18-1128."
    ),
    "sklearn2026averageprecision": (
        'scikit-learn developers, "average_precision_score API," scikit-learn API '
        "Reference, 2026. [Online]. Available: https://scikit-learn.org/stable/modules/"
        "generated/sklearn.metrics.average_precision_score.html. Accessed: Aug. 5, 2026."
    ),
    "sokolova2009metrics": (
        'M. Sokolova and G. Lapalme, "A Systematic Analysis of Performance Measures for '
        'Classification Tasks," Inf. Process. Manage., vol. 45, no. 4, pp. 427–437, 2009, '
        "doi: 10.1016/j.ipm.2009.03.002."
    ),
    "guo2017calibration": (
        'C. Guo, G. Pleiss, Y. Sun, et al., "On Calibration of Modern Neural Networks," '
        "in Proc. ICML, vol. 70, 2017, pp. 1321–1330. [Online]. Available: "
        "https://proceedings.mlr.press/v70/guo17a.html"
    ),
    "cawley2010selection": (
        'G. C. Cawley and N. L. C. Talbot, "On Over-Fitting in Model Selection and '
        'Subsequent Selection Bias in Performance Evaluation," J. Mach. Learn. Res., '
        "vol. 11, pp. 2079–2107, 2010."
    ),
    "banko2020taxonomy": (
        'M. Banko, B. MacKeen, and L. Ray, "A Unified Taxonomy of Harmful Content," in '
        "Proc. 4th Workshop Online Abuse and Harms, 2020, pp. 125–137, "
        "doi: 10.18653/v1/2020.alw-1.16."
    ),
    "waseem2017abuse": (
        'Z. Waseem, T. Davidson, D. Warmsley, et al., "Understanding Abuse: A Typology of '
        'Abusive Language Detection Subtasks," in Proc. 1st Workshop Abusive Language '
        "Online, 2017, pp. 78–84, doi: 10.18653/v1/W17-3012."
    ),
    "elsherief2021implicit": (
        'M. ElSherief, C. Ziems, D. Muchlinski, et al., "Latent Hatred: A Benchmark for '
        'Understanding Implicit Hate Speech," in Proc. EMNLP, 2021, pp. 345–363, '
        "doi: 10.18653/v1/2021.emnlp-main.29."
    ),
    "ilic2018irony": (
        'S. Ilić, E. Marrese-Taylor, J. Balazs, et al., "Deep Contextualized Word '
        'Representations for Detecting Sarcasm and Irony," in Proc. WASSA, 2018, pp. 2–7, '
        "doi: 10.18653/v1/W18-6202."
    ),
    "almeida2022motoso": (
        'V. Zavala and C. Almeida, "‘Motoso y terruco’: ideologías lingüísticas y '
        'racialización en la política peruana," Lexis, vol. 46, no. 2, pp. 481–521, 2022, '
        "doi: 10.18800/lexis.202202.002."
    ),
    "albornoz2018conocer": (
        'D. Albornoz and M. Flores, "Conocer para resistir: violencia de género en línea '
        'en Perú," Hiperderecho, Lima, Perú, 2018. [Online]. Available: '
        "https://hiperderecho.org/tecnoresistencias/wp-content/uploads/2019/01/"
        "violencia_genero_linea_peru_2018.pdf"
    ),
    "defensoria2021violenciaenlinea": (
        'Defensoría del Pueblo del Perú, "Violencia de género contra las mujeres en '
        'línea," Documento de Trabajo no. 001-2021-DP/ADM, Aug. 2021. [Online]. Available: '
        "https://www.defensoria.gob.pe/wp-content/uploads/2021/08/Documento-de-trabajo-01-"
        "Violencia-de-g%C3%A9nero-contra-las-mujeres-en-l%C3%ADnea.pdf"
    ),
    "lovon2022lesbofobia": (
        'C. M. Lovón-Cueva and M. Lovón-Cueva, "Lesbian Lexicon: The Construction of a '
        'Repertoire of Hate in Peruvian Cyberforums," Whatever, vol. 5, no. 1, pp. 43–70, '
        "2022, doi: 10.13131/2611-657X.whatever.v5i1.156."
    ),
    "youtube2026sexualpolicy": (
        'YouTube, "Política sobre desnudos y contenido sexual," Ayuda de YouTube, 2026. '
        "[Online]. Available: https://support.google.com/youtube/answer/2802002?hl=es-419. "
        "Accessed: Aug. 5, 2026."
    ),
    "chow1970reject": (
        'C. K. Chow, "On Optimum Recognition Error and Reject Tradeoff," IEEE Trans. Inf. '
        "Theory, vol. 16, no. 1, pp. 41–46, 1970, doi: 10.1109/TIT.1970.1054406."
    ),
    "geifman2017selective": (
        'Y. Geifman and R. El-Yaniv, "Selective Classification for Deep Neural Networks," '
        "in Adv. Neural Inf. Process. Syst., vol. 30, 2017."
    ),
    "mozannar2020defer": (
        'H. Mozannar and D. Sontag, "Consistent Estimators for Learning to Defer to an '
        'Expert," in Proc. ICML, vol. 119, 2020, pp. 7076–7087. [Online]. Available: '
        "https://proceedings.mlr.press/v119/mozannar20b.html"
    ),
    "gorwa2020moderation": (
        'R. Gorwa, R. Binns, and C. Katzenbach, "Algorithmic Content Moderation: Technical '
        'and Political Challenges in the Automation of Platform Governance," Big Data & '
        "Society, vol. 7, no. 1, 2020, doi: 10.1177/2053951719897945."
    ),
    "andersen2021rem": (
        'J. S. Andersen, O. Zukunft, and W. Maalej, "REM: Efficient Semi-Automated '
        'Real-Time Moderation of Online Forums," in Proc. ACL-IJCNLP: System Demonstrations, '
        "2021, pp. 142–149, doi: 10.18653/v1/2021.acl-demo.17."
    ),
}


def apply_citations(cells: Iterable[Any], metadata: dict[str, Any]) -> Any:
    """Numera marcadores por primera aparición y crea la bibliografía final."""

    ordered_keys: list[str] = []
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        for key in CITATION_MARKER.findall(cell.source):
            if key not in REFERENCE_LIBRARY:
                raise KeyError(f"Referencia de cuaderno no registrada: {key}")
            if key not in ordered_keys:
                ordered_keys.append(key)
    if not ordered_keys:
        raise ValueError(
            "Cada cuaderno activo debe contener al menos una cita académica"
        )
    numbers = {key: index for index, key in enumerate(ordered_keys, start=1)}
    for cell in cells:
        if cell.cell_type == "markdown":
            cell.source = CITATION_MARKER.sub(
                lambda match: f"[{numbers[match.group(1)]}]", cell.source
            )
    metadata["citation_style"] = "IEEE_numeric"
    metadata["citation_keys"] = ordered_keys
    metadata["reference_count"] = len(ordered_keys)
    bibliography = "## Referencias\n\n" + "\n\n".join(
        f"[{numbers[key]}] {REFERENCE_LIBRARY[key]}" for key in ordered_keys
    )
    return bibliography
