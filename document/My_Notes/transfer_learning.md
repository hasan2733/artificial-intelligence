# Transfer Learning & Popular Pretrained Models

## Table of Contents
1. [What is Transfer Learning?](#what-is-transfer-learning)
2. [Why Transfer Learning?](#why-transfer-learning)
3. [How Transfer Learning Works](#how-transfer-learning-works)
4. [Types of Transfer Learning](#types-of-transfer-learning)
5. [Popular Pretrained Models](#popular-pretrained-models)
   - [Computer Vision](#computer-vision-models)
   - [Natural Language Processing](#nlp-models)
   - [Multimodal Models](#multimodal-models)
6. [Choosing the Right Pretrained Model](#choosing-the-right-pretrained-model)
7. [Best Practices](#best-practices)

---

## What is Transfer Learning?

**Transfer Learning** is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different (but related) task. Instead of training a neural network from scratch, you take a model that has already learned useful features from a large dataset and adapt it to your specific problem.

> **Analogy:** It's like a person who learned to play the piano and then picks up a guitar — they don't start from zero. Their understanding of rhythm, music theory, and finger coordination transfers and accelerates learning the new instrument.

---

## Why Transfer Learning?

| Challenge (From Scratch) | Solution (Transfer Learning) |
|---|---|
| Requires millions of labeled samples | Works well with small datasets |
| Weeks/months of training time | Fine-tuning takes hours or days |
| Needs expensive GPU clusters | Can fine-tune on a single GPU |
| High risk of overfitting on small data | Pretrained weights act as regularization |
| Requires deep ML expertise to design architecture | Use proven, state-of-the-art architectures |

---

## How Transfer Learning Works

```
┌─────────────────────────────────────────────────────┐
│              PRETRAINED MODEL                       │
│                                                     │
│  [Input] → [Feature Extractor Layers] → [Head]     │
│             (frozen or fine-tuned)       (replaced) │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              YOUR CUSTOM MODEL                      │
│                                                     │
│  [Input] → [Reused Layers] → [New Task-Specific     │
│                                Classification Head] │
└─────────────────────────────────────────────────────┘
```

### Steps Involved:

1. **Select** a pretrained model relevant to your domain
2. **Load** the model with its pretrained weights
3. **Freeze** base layers (optional — prevents overwriting learned features)
4. **Replace** the final layer(s) with a new head for your task
5. **Fine-tune** on your dataset with a low learning rate

---

## Types of Transfer Learning

### 1. Feature Extraction
- Freeze all pretrained layers
- Only train the new classification head
- Best when: your dataset is small and similar to the pretraining data

### 2. Fine-Tuning
- Unfreeze some or all pretrained layers
- Train end-to-end on your data (with a low learning rate)
- Best when: your dataset is large enough, or your domain differs from pretraining

### 3. Domain Adaptation
- Adapting a model from one domain (e.g., news articles) to another (e.g., medical records)
- May involve intermediate pretraining steps

### 4. Zero-Shot / Few-Shot Transfer
- Using a large pretrained model to handle tasks it was never explicitly trained on
- Common in modern LLMs (e.g., GPT-4, Claude)

---

## Popular Pretrained Models

---

### Computer Vision Models

#### 1. VGG16 / VGG19
| Property | Details |
|---|---|
| **Developed by** | Visual Geometry Group, Oxford (2014) |
| **Training Dataset** | ImageNet (1.2M images, 1,000 classes) |
| **Architecture** | Deep CNN with 16/19 weight layers |
| **Parameters** | ~138M (VGG16) |
| **Input Size** | 224 × 224 RGB |
| **Common Use Cases** | Image classification, feature extraction |

**Strengths:** Simple, uniform architecture; easy to understand.  
**Weaknesses:** Very large model size; slow inference.

---

#### 2. ResNet (Residual Networks)
| Property | Details |
|---|---|
| **Developed by** | Microsoft Research (2015) |
| **Variants** | ResNet-18, 34, 50, 101, 152 |
| **Training Dataset** | ImageNet (1.2M images, 1,000 classes) |
| **Key Innovation** | Skip connections (residual blocks) — solves vanishing gradient |
| **Parameters** | 11M (ResNet-18) to 60M (ResNet-152) |
| **Common Use Cases** | Classification, object detection backbone, medical imaging |

**Strengths:** Deep networks without degradation; widely adopted.  
**Weaknesses:** Can still be computationally heavy in deeper variants.

---

#### 3. EfficientNet
| Property | Details |
|---|---|
| **Developed by** | Google Brain (2019) |
| **Variants** | EfficientNet-B0 to B7 |
| **Training Dataset** | ImageNet; larger variants also use JFT-300M (300M images) |
| **Key Innovation** | Compound scaling — scales width, depth, and resolution together |
| **Parameters** | 5.3M (B0) to 66M (B7) |
| **Common Use Cases** | Mobile/edge deployment, high-accuracy classification |

**Strengths:** Best accuracy-to-efficiency ratio; scalable.  
**Weaknesses:** Complex training; sensitive to hyperparameters.

---

#### 4. Vision Transformer (ViT)
| Property | Details |
|---|---|
| **Developed by** | Google Brain (2020) |
| **Variants** | ViT-B/16, ViT-L/16, ViT-H/14 |
| **Training Dataset** | JFT-300M (300M images); also ImageNet-21K (14M images, 21K classes) |
| **Key Innovation** | Applies Transformer architecture (self-attention) to image patches |
| **Parameters** | 86M (ViT-B) to 632M (ViT-H) |
| **Common Use Cases** | Image classification, backbone for detection/segmentation |

**Strengths:** Excellent scalability; captures global context.  
**Weaknesses:** Requires large datasets to train well; high compute.

---

#### 5. YOLO (You Only Look Once)
| Property | Details |
|---|---|
| **Developed by** | Joseph Redmon et al. (2016); now YOLOv8/v9 by Ultralytics |
| **Training Dataset** | COCO (330K images, 80 object categories), ImageNet |
| **Key Innovation** | Single-pass object detection — extremely fast |
| **Common Use Cases** | Real-time object detection, video surveillance, robotics |

**Strengths:** Real-time performance; easy to deploy.  
**Weaknesses:** Lower accuracy on small objects compared to two-stage detectors.

---

### NLP Models

#### 6. BERT (Bidirectional Encoder Representations from Transformers)
| Property | Details |
|---|---|
| **Developed by** | Google AI (2018) |
| **Variants** | BERT-Base (110M params), BERT-Large (340M params), RoBERTa, DistilBERT, AlBERT |
| **Training Dataset** | BooksCorpus (800M words) + English Wikipedia (2,500M words) |
| **Training Objective** | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) |
| **Common Use Cases** | Text classification, NER, question answering, sentiment analysis |

**Strengths:** Deep bidirectional context understanding; excellent for NLU tasks.  
**Weaknesses:** Encoder-only; not suited for text generation.

---

#### 7. GPT Series (Generative Pretrained Transformer)
| Property | Details |
|---|---|
| **Developed by** | OpenAI |
| **Versions** | GPT-2 (2019), GPT-3 (2020), GPT-4 (2023) |
| **Training Dataset** | WebText (40GB web scraped) for GPT-2; Common Crawl, Books, Wikipedia, code for GPT-3/4 (trillions of tokens) |
| **Parameters** | 117M (GPT-2 small) → ~1.76T estimated (GPT-4, mixture of experts) |
| **Common Use Cases** | Text generation, summarization, code generation, chatbots |

**Strengths:** State-of-the-art text generation; powerful few-shot learner.  
**Weaknesses:** Very large and expensive; can hallucinate.

---

#### 8. T5 (Text-to-Text Transfer Transformer)
| Property | Details |
|---|---|
| **Developed by** | Google Research (2019) |
| **Variants** | T5-Small to T5-11B; Flan-T5 |
| **Training Dataset** | C4 (Colossal Clean Crawled Corpus — 750GB of English web text) |
| **Key Innovation** | Frames every NLP task as text-to-text generation |
| **Parameters** | 60M (Small) to 11B (XXL) |
| **Common Use Cases** | Translation, summarization, Q&A, classification |

**Strengths:** Unified framework for all NLP tasks; flexible and powerful.  
**Weaknesses:** Slower inference; needs significant compute for large variants.

---

#### 9. LLaMA / LLaMA 2 / LLaMA 3
| Property | Details |
|---|---|
| **Developed by** | Meta AI (2023–2024) |
| **Variants** | LLaMA 3: 8B, 70B, 405B parameters |
| **Training Dataset** | Over 15 trillion tokens from publicly available sources (web, code, scientific papers) |
| **License** | Open weights (community use, research, and commercial with restrictions) |
| **Common Use Cases** | Open-source LLM fine-tuning, local deployment, research |

**Strengths:** Open-weight; highly capable; great community support.  
**Weaknesses:** Large models still require significant hardware.

---

### Multimodal Models

#### 10. CLIP (Contrastive Language–Image Pretraining)
| Property | Details |
|---|---|
| **Developed by** | OpenAI (2021) |
| **Training Dataset** | 400 million (image, text) pairs scraped from the internet |
| **Key Innovation** | Learns joint image-text embeddings via contrastive learning |
| **Common Use Cases** | Zero-shot image classification, image search, image captioning backbone |

**Strengths:** Zero-shot generalization; bridges vision and language.  
**Weaknesses:** Limited fine-grained understanding; struggles with counting, spatial reasoning.

---

#### 11. Whisper
| Property | Details |
|---|---|
| **Developed by** | OpenAI (2022) |
| **Training Dataset** | 680,000 hours of multilingual audio from the web |
| **Variants** | Tiny, Base, Small, Medium, Large |
| **Common Use Cases** | Automatic speech recognition (ASR), translation, transcription |

**Strengths:** Multilingual; robust to accents and noise; open source.  
**Weaknesses:** Slower than specialized ASR systems; large model size.

---

## Choosing the Right Pretrained Model

```
Is your task visual?
├── Yes → Image Classification?    → ResNet / EfficientNet / ViT
│         Object Detection?        → YOLOv8 / Faster R-CNN
│         Vision + Language?       → CLIP / Florence
│
└── No  → Text Understanding?      → BERT / RoBERTa
          Text Generation?         → GPT / LLaMA / T5
          Speech?                  → Whisper
          Multilingual?            → mBERT / XLM-R
```

### Decision Factors:
- **Dataset size:** Small → Feature extraction. Large → Full fine-tuning.
- **Domain similarity:** Close to ImageNet/Wikipedia → Less fine-tuning needed.
- **Inference speed:** Edge/mobile → MobileNet, DistilBERT, EfficientNet-B0.
- **Accuracy priority:** Cloud/server → ViT-Large, GPT-4, LLaMA-70B.
- **Open source needed:** LLaMA, Whisper, BERT, T5 are fully open.

---

## Best Practices

### 1. Learning Rate
- Use a **small learning rate** (e.g., `1e-4` to `1e-5`) when fine-tuning pretrained models to avoid destroying learned representations.

### 2. Layer Freezing Strategy
- **Freeze all** → train only new head (small dataset)
- **Unfreeze top layers** → train head + last few blocks (medium dataset)
- **Unfreeze all** → full fine-tuning (large dataset, similar domain)

### 3. Data Augmentation
- Apply aggressive augmentation when your dataset is small to reduce overfitting.

### 4. Batch Normalization
- When fine-tuning, consider **keeping BatchNorm layers frozen** to preserve learned statistics.

### 5. Early Stopping
- Monitor validation loss; stop training when it stops improving to prevent overfitting.

### 6. Use Mixed Precision
- Train with `float16` (AMP) to speed up fine-tuning and reduce GPU memory usage.

---

## Summary Table

| Model | Domain | Training Data | Parameters | Open Source |
|---|---|---|---|---|
| VGG16 | Vision | ImageNet (1.2M) | 138M | ✅ |
| ResNet-50 | Vision | ImageNet (1.2M) | 25M | ✅ |
| EfficientNet-B0 | Vision | ImageNet / JFT | 5.3M | ✅ |
| ViT-B/16 | Vision | ImageNet-21K / JFT | 86M | ✅ |
| YOLOv8 | Detection | COCO | ~3–68M | ✅ |
| BERT-Base | NLP | BooksCorpus + Wiki | 110M | ✅ |
| GPT-3 | NLP | Common Crawl + more | 175B | ❌ |
| T5-Base | NLP | C4 (750GB) | 220M | ✅ |
| LLaMA 3-8B | NLP | 15T tokens | 8B | ✅ |
| CLIP | Multimodal | 400M image-text pairs | ~430M | ✅ |
| Whisper Large | Audio | 680K hours audio | 1.55B | ✅ |

---

*Last updated: May 2026*
