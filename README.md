# Hoja de trabajo #2: Transformers y mecanismos de atención

**CC3092 – Deep Learning y Sistemas Inteligentes**

Implementación, verificación y visualización del mecanismo de atención: un ejercicio a mano verificado con NumPy y PyTorch, el efecto de dividir entre √d_k, el costo computacional de la atención y el análisis de los pesos de atención de BERT y GPT-2.

## Contenido

| Archivo | Descripción |
|---|---|
| `Hoja2_Transformers_Atencion.ipynb` | Notebook ejecutado y comentado con toda la implementación y las visualizaciones. |
| `Hoja2_Transformers_Atencion.docx` | Informe: investigación, ejercicio a mano, verificaciones, mapas de atención, discusión, conclusiones y referencias. |
| `Hoja_de_trabajo_2_Transformers_Atencion-1.pdf` | Enunciado de la hoja de trabajo. |
| `figs/` | Figuras que genera el notebook y que se usan en el informe. |

## Secciones del notebook

0. **Configuración**
1. **Ejercicio a mano.** Atención con n = 3 y d_k = 2, calculada en NumPy y comparada con `F.scaled_dot_product_attention`, con y sin máscara causal.
2. **Verificación de √d_k.** Muestra por simulación que Var(q·k) ≈ d_k y que, sin escalar, la softmax se satura y su gradiente se desvanece.
3. **Atención en PyTorch.** Ejemplos de `F.scaled_dot_product_attention`, `nn.MultiheadAttention` (máscaras, pesos por cabeza, `kdim`/`vdim`), `nn.TransformerEncoderLayer` (Post-LN y Pre-LN) y `generate_square_subsequent_mask`.
4. **Costo computacional.** Mide el tiempo y la memoria de la atención en función de la longitud n.
5. **Atención en BERT y GPT-2** (`bert-base-uncased` y `gpt2`)
   - 5.1 Mapas de calor de 4 cabezas de capas distintas y detección automática de patrones en las 144 cabezas.
   - 5.2 Par «…too tired» / «…too wide»: búsqueda de cabezas en las que «it» atiende a «animal» o a «street».
   - 5.3 GPT-2: comprobación de que la atención es triangular inferior y análisis del *attention sink* (Xiao et al., 2023).
   - 5.4 Entropía promedio de la atención por capa en BERT y GPT-2.
   - 5.5 (Opcional) bertviz, que está comentado.

## Resultados principales

- **Ejercicio a mano.** Coincide con NumPy y con PyTorch; la diferencia máxima es de ~9·10⁻¹⁶.
- **√d_k.** Var(q·k) ≈ d_k para d_k entre 1 y 1024. Sin escalar y con d_k = 1024, la probabilidad máxima de la softmax llega a ≈ 0.97 y la norma del jacobiano cae de 0.33 a 0.05.
- **Costo.** El tiempo crece de forma cuadrática con n. Con n = 100,000, la matriz de atención de una sola cabeza en fp32 ocuparía ≈ 37 GiB.
- **BERT.** Hay cabezas especializadas en el token anterior, el token siguiente, `[CLS]`, `[SEP]` y la puntuación. En la capa 6, cabeza 10, «it» atiende más a «animal» (0.28) con *tired* y a «street» (0.59) con *wide*.
- **GPT-2.** La parte triangular superior vale exactamente 0. El primer token recibe entre 23 % y 83 % de la atención según la capa, frente a ≈ 17 % si la atención fuera uniforme (*attention sink*).
- **Entropía.** En ambos modelos, las capas intermedias y profundas atienden de forma más concentrada que las primeras, y GPT-2 de forma más concentrada que BERT.

## Cómo ejecutarlo

Requiere Python 3.10 o superior y una conexión a internet la primera vez, para descargar los modelos (~1 GB).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers matplotlib numpy jupyter
jupyter notebook Hoja2_Transformers_Atencion.ipynb
```

En Google Colab basta con descomentar la línea `!pip install ...` de la primera celda.

Las figuras se guardan en `figs/`. En CPU, los tiempos de la sección 4 cambian entre corridas.

**Si falla la descarga de Hugging Face con `CERTIFICATE_VERIFY_FAILED`** (pasa en redes o con antivirus que interceptan TLS), descarga los modelos una vez usando el almacén de certificados del sistema:

```bash
pip install truststore
python -c "import truststore; truststore.inject_into_ssl(); from transformers import AutoModel, AutoTokenizer; [(AutoTokenizer.from_pretrained(m), AutoModel.from_pretrained(m)) for m in ('bert-base-uncased', 'gpt2')]"
```

Después ejecuta el notebook con la variable de entorno `HF_HUB_OFFLINE=1` para que lea los modelos de la caché local.

## Referencias principales

- Vaswani et al. (2017). *Attention Is All You Need.* arXiv:1706.03762
- Bahdanau et al. (2015); Luong et al. (2015)
- Xiao et al. (2023). *Efficient Streaming Language Models with Attention Sinks.* arXiv:2309.17453
- Jain & Wallace (2019). *Attention is not Explanation*; Wiegreffe & Pinter (2019). *Attention is not not Explanation*

La lista completa de referencias está en el informe.
