# Information Retrieval System

نظام استرجاع معلومات (Information Retrieval System) مبني وفق **Clean Architecture**، يدعم الفهرسة الكاملة لمجموعتي **Quora** و **Webis-Touché 2020 v2**، وخمسة نماذج استرجاع، والتقييم باستخدام `qrels`، وواجهة Streamlit، وواجهة FastAPI.

هذا الملف يشرح التشغيل من الصفر، بناء الفهارس، استكمال البناء بعد الانقطاع، البحث، التقييم، الرسومات، Query Refinement، وDocument Clustering.

---

## 1. المزايا

- Streaming لقراءة الوثائق دون تحميل Corpus كاملاً في RAM.
- Batch Processing للفهرسة على أجهزة بذاكرة محدودة.
- فهرس نصي كامل على SQLite يخدم BM25 وTF-IDF.
- فهرس دلالي كامل باستخدام Sentence Transformers وFAISS.
- Checkpoint وResume أثناء بناء الفهارس الطويلة.
- خمسة نماذج: TF-IDF، BM25، Embedding، Hybrid Serial، Hybrid Parallel.
- Query Refinement اختياري وقابل للتشغيل والإيقاف.
- Document Clustering باستخدام MiniBatchKMeans.
- تقييم الاسترجاع بـMAP@10 وnDCG@10 وRecall@10 وPrecision@10.
- تقييم Clustering بـSilhouette وDavies-Bouldin وInertia ورسمي PCA وCluster Sizes.
- Streamlit وFastAPI وSwagger.
- اختبارات آلية؛ العدد الحالي المتوقع `32 passed`.

---

## 2. مجموعات البيانات

| Dataset | الاسم في الأوامر | المهمة | الوثائق | Queries | qrels |
|---|---|---|---:|---:|---:|
| Quora | `quora` | Duplicate-question retrieval | 522,931 | 10,000 | 15,675 |
| Webis-Touché 2020 v2 | `touche2020-v2` | Argument retrieval | 382,545 | 49 | 2,214 |

- Quora تستخدم `processing_profile=question`.
- Touché تستخدم `processing_profile=argument` وتحتوي `title` و`stance` و`url`.
- Natural Questions إعداد قديم (legacy) وليست من مجموعات التسليم النهائية.

---

## 3. بنية المشروع

```text
app/
  domain/          الكيانات والقواعد الأساسية
  application/     الخدمات وتنسيق العمليات
  infrastructure/  البيانات والمعالجة والفهارس والخوارزميات والتقييم
  presentation/    Streamlit وFastAPI
scripts/           أوامر الفهرسة والبحث والتقييم والرسومات
tests/             الاختبارات الآلية
storage/           الفهارس والنتائج المحلية
docs/              التوثيق
```

قاعدة مختصرة:

```text
Domain يعرّف
Application ينسّق
Infrastructure ينفّذ
Presentation يعرض
```

---

# التشغيل من الصفر

## 4. المتطلبات المقترحة

تم اختبار المشروع على Windows وPython 3.13 وجهاز CPU مع 8 GB RAM وSSD. لا يلزم GPU منفصل.

- الحد الأدنى العملي للمساحة الحرة: 10 GB.
- الأفضل: 15-20 GB.
- أثناء بناء الفهارس: وصّل الجهاز بالشاحن وعطّل Sleep مؤقتاً.

## 5. تنزيل المشروع وإنشاء البيئة

```bash
git clone https://github.com/Hasan-Alhendi/information-retrieval-system.git
cd information-retrieval-system
python -m venv .env
```

Command Prompt:

```bat
.env\Scripts\activate
```

PowerShell:

```powershell
.env\Scripts\Activate.ps1
```

تثبيت المكتبات:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

قد يستغرق التثبيت وتنزيل النماذج `10-30 دقيقة` أو أكثر حسب الإنترنت. قد يظهر تحذير HF Hub للطلبات غير الموثقة؛ لا يمنع التشغيل.

## 6. تشغيل الاختبارات

```bash
pytest
```

النتيجة المتوقعة في النسخة النهائية:

```text
32 passed
```

العدد 32 يتكون من 29 اختباراً أساسياً، إضافة إلى 3 اختبارات لمقاييس Clustering وPCA وحالات الحافة.

---

# التحقق والاختبارات المصغرة

## 7. التحقق من البيانات

```bash
python scripts/validate_full_dataset.py --dataset quora
python scripts/validate_full_dataset.py --dataset touche2020-v2
```

قد يستغرق التنزيل والتحقق عدة دقائق إلى نصف ساعة حسب الإنترنت وCache.

## 8. Smoke Test للفهرس النصي

```bash
python scripts/smoke_test_full_index.py --dataset quora
python scripts/smoke_test_full_index.py --dataset touche2020-v2
```

عادة من دقيقة إلى عدة دقائق.

## 9. Smoke Test لفهرس Embedding

```bash
python scripts/smoke_test_full_embedding.py --dataset quora --checkpoint-size 1000 --force
python scripts/smoke_test_full_embedding.py --dataset touche2020-v2 --checkpoint-size 1000 --force
```

عادة `2-10 دقائق`. استخدام `--force` هنا آمن لأن Smoke Test منفصل. لا تستخدمه مع الفهرس الكامل إلا لإعادة البناء من الصفر.

---

# بناء الفهارس الكاملة

## 10. قواعد مهمة

1. الفهرس النصي الواحد يخدم BM25 وTF-IDF معاً.
2. فهرس Embedding مستقل ويستخدم FAISS.
3. ابنِ `lexical` ثم `embedding` بشكل منفصل على جهاز 8 GB RAM.
4. عند الانقطاع أعد الأمر نفسه دون `--force` ليستكمل من آخر Checkpoint.
5. `--force` يحذف الفهرس المحدد ويبدأ من الصفر.

أماكن الفهارس:

```text
storage/indexes/<dataset>/disk_lexical/full/lexical.sqlite3
storage/vector_stores/<dataset>/embedding_sentence-transformers__all-MiniLM-L6-v2/full/
```

## 11. Quora كاملة

```bash
python scripts/build_full_dataset.py --dataset quora --components lexical
python scripts/build_full_dataset.py --dataset quora --components embedding
```

| العملية | الزمن التقريبي على CPU و8 GB RAM وSSD |
|---|---:|
| Quora Lexical | 30-90 دقيقة، وقد يزيد |
| Quora Embedding | 35-90 دقيقة |

## 12. Touché كاملة

```bash
python scripts/build_full_dataset.py --dataset touche2020-v2 --components lexical
python scripts/build_full_dataset.py --dataset touche2020-v2 --components embedding
```

| العملية | الزمن التقريبي |
|---|---:|
| Touché Lexical | ساعة إلى 3 ساعات، وقد يتجاوز ذلك |
| Touché Embedding | 45 دقيقة إلى ساعتين |

نصوص Touché أطول، لذلك قد يستغرق بناؤها زمناً أطول رغم أن عدد وثائقها أقل.

## 13. الاستكمال بعد الانقطاع

أعد الأمر نفسه دون `--force`، مثلاً:

```bash
python scripts/build_full_dataset.py --dataset quora --components embedding
```

يعتمد Resume على `processed_documents` وCheckpoints وMetadata وتطابق عدد صفوف FAISS.

## 14. فحص الحالة

```bash
python scripts/full_dataset_status.py --dataset quora
python scripts/full_dataset_status.py --dataset touche2020-v2
```

الحالة المطلوبة:

```text
lexical.finalized = true
embedding.finalized = true
ready_for_all_models = true
```

## 15. إصلاح SQLite عند الحاجة

إذا ظهر:

```text
sqlite3.DatabaseError: database disk image is malformed
```

لا تستخدم `--force` مباشرة.

```bash
python scripts/repair_lexical_index.py --dataset quora --diagnose-only
python scripts/repair_lexical_index.py --dataset quora --batch-size 256
```

Touché:

```bash
python scripts/repair_lexical_index.py --dataset touche2020-v2 --diagnose-only
python scripts/repair_lexical_index.py --dataset touche2020-v2 --batch-size 128
```

---

# البحث

## 16. BM25 وTF-IDF

```bash
python scripts/search_full.py --dataset quora --model bm25 --query "How can I learn programming?" --top-k 10
python scripts/search_full.py --dataset quora --model tfidf --query "How can I learn programming?" --top-k 10
```

```bash
python scripts/search_full.py --dataset touche2020-v2 --model bm25 --query "Should teachers get tenure?" --top-k 10
python scripts/search_full.py --dataset touche2020-v2 --model tfidf --query "Should teachers get tenure?" --top-k 10
```

## 17. Embedding

```bash
python scripts/search_full_embedding.py --dataset quora --query "How can I learn programming?" --top-k 10 --batch-size 32
python scripts/search_full_embedding.py --dataset touche2020-v2 --query "Should teachers get tenure?" --top-k 10 --batch-size 16
```

أول استعلام في عملية Python جديدة قد يستغرق `8-15 ثانية` بسبب تحميل Sentence Transformer وFAISS. بعد Warm-up تصبح الاستعلامات اللاحقة أسرع بكثير.

## 18. Hybrid

```bash
python scripts/search_full_hybrid.py --dataset quora --model hybrid_serial --query "How can I learn programming?" --top-k 10
python scripts/search_full_hybrid.py --dataset quora --model hybrid_parallel --query "How can I learn programming?" --top-k 10
```

```bash
python scripts/search_full_hybrid.py --dataset touche2020-v2 --model hybrid_serial --query "Should teachers get tenure?" --top-k 10
python scripts/search_full_hybrid.py --dataset touche2020-v2 --model hybrid_parallel --query "Should teachers get tenure?" --top-k 10
```

Hybrid Parallel أبطأ عادة لأنه يشغل TF-IDF وBM25 وEmbedding ثم يدمج الترتيب باستخدام RRF.

---

# الواجهات

## 19. Streamlit

```bash
streamlit run app/presentation/streamlit/ui.py
```

افتح:

```text
http://localhost:8501
```

التبويبات: Search، Evaluation، Datasets، Clustering.

## 20. FastAPI

```bash
uvicorn app.main:app --reload
```

افتح:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

مثال `POST /search` للفهرس الكامل:

```json
{
  "query": "How can I learn programming?",
  "dataset_name": "quora",
  "model_name": "embedding",
  "top_k": 10,
  "max_docs": null,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "use_query_refinement": false
}
```

---

# التقييم والرسومات

## 21. تقييم سريع

```bash
python scripts/evaluate_full_system.py --dataset quora --max-queries 100 --top-k 10
python scripts/evaluate_full_system.py --dataset touche2020-v2 --max-queries 10 --top-k 10
```

## 22. التقييم الرسمي الكامل

```bash
python scripts/evaluate_full_system.py --dataset quora --max-queries 10000 --top-k 10
python scripts/evaluate_full_system.py --dataset touche2020-v2 --max-queries 49 --top-k 10
```

| العملية | الزمن التقريبي |
|---|---:|
| Quora، 5 نماذج، 10,000 Query | ساعة إلى 4 ساعات، وقد يزيد |
| Touché، 5 نماذج، 49 Query | عدة دقائق إلى نحو 20 دقيقة |

## 23. Benchmark السرعة

```bash
python scripts/benchmark_full_search.py --dataset quora --models bm25 tfidf --max-queries 10 --repeats 3 --warmups 1 --top-k 10
python scripts/benchmark_dense_hybrids.py --dataset quora --models embedding hybrid_serial hybrid_parallel --max-queries 10 --repeats 3 --top-k 10
```

استخدم الأوامر نفسها مع `touche2020-v2`.

## 24. إنشاء الرسومات

```bash
python scripts/generate_evaluation_charts.py --dataset quora
python scripts/generate_evaluation_charts.py --dataset touche2020-v2
```

تحفظ في:

```text
storage/evaluation/charts/quora/
storage/evaluation/charts/touche2020-v2/
```

---

# النتائج المرجعية النهائية

## 25. Quora - جميع 10,000 Query

| Model | MAP@10 | Recall@10 | Precision@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.6565 | 0.8116 | 0.1086 | 0.7073 |
| BM25 | 0.6898 | 0.8434 | 0.1135 | 0.7399 |
| Embedding | **0.8363** | **0.9503** | **0.1337** | **0.8755** |
| Hybrid Serial | 0.8153 | 0.9279 | 0.1291 | 0.8548 |
| Hybrid Parallel | 0.7549 | 0.8875 | 0.1214 | 0.8003 |

أفضل نموذج: Embedding.

## 26. Touché - جميع 49 Query

| Model | MAP@10 | Recall@10 | Precision@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.0563 | 0.1183 | 0.1735 | 0.1591 |
| BM25 | **0.1121** | 0.1892 | 0.2673 | **0.2861** |
| Embedding | 0.0758 | 0.1443 | 0.1857 | 0.1886 |
| Hybrid Serial | 0.1112 | **0.1902** | **0.2714** | 0.2826 |
| Hybrid Parallel | 0.0994 | 0.1742 | 0.2510 | 0.2531 |

أفضل MAP وnDCG: BM25. أفضل Recall وPrecision: Hybrid Serial.

---

# الميزات الإضافية

## 27. Query Refinement

الميزة اختيارية وتتضمن Normalization، قاموس تصحيح Rule-based محدود، وQuery Expansion.

مثال:

```text
infomation retrival querry
```

يصبح:

```text
information retrieval query search ranking question
```

هذه ليست أداة تصحيح إنجليزية عامة؛ تصحح الكلمات الموجودة في القاموس فقط.

## 28. Document Clustering

تشغيل مستقل وحفظ النتائج:

```bash
python scripts/evaluate_clustering.py --dataset quora --max-docs 1000 --clusters 5 --sample-documents 3
```

تحفظ الملفات في:

```text
storage/evaluation/clustering/quora/
  clustering_1000_docs_5_clusters.json
  cluster_sizes.png
  pca_projection.png
```

الإعداد المرجعي:

- Dataset: Quora
- Documents: 1,000
- Clusters: 5
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Algorithm: MiniBatchKMeans

النتائج:

| Metric | Value |
|---|---:|
| Silhouette Score | 0.063989 |
| Davies-Bouldin Index | 3.816218 |
| Inertia | 815.805786 |

أحجام العناقيد: `206، 130، 94، 443، 127`.

قراءة النتيجة: التنفيذ ناجح، لكن يوجد تداخل ملحوظ بين بعض العناقيد. PCA للعرض فقط؛ التجميع الحقيقي يتم على المتجهات الأصلية ذات 384 بُعداً.

---

# حل المشكلات

## 29. بطء أول استعلام

Cold Start بسبب تحميل النموذج وFAISS وMetadata. استخدم Streamlit أو FastAPI للاحتفاظ بالمكونات في الذاكرة.

## 30. نفاد RAM

- استخدم أوامر Full Dataset المعتمدة على Streaming.
- قلل Batch Size:

```bash
python scripts/build_full_dataset.py --dataset quora --components embedding --embedding-batch-size 16
```

- لا تشغل Build وEvaluation وStreamlit معاً على جهاز 8 GB RAM.

## 31. نفاد مساحة القرص

لا تحذف مجلدات `full` إلا إذا كنت تريد إعادة الفهرسة. افحص الحالة والحجم باستخدام `full_dataset_status.py`.

---

# التسلسل المختصر

```text
1. git clone
2. virtual environment
3. pip install -r requirements.txt
4. pytest
5. validate_full_dataset
6. smoke tests
7. build lexical
8. build embedding
9. full_dataset_status
10. search
11. evaluate_full_system
12. benchmarks
13. generate_evaluation_charts
14. evaluate_clustering
15. Streamlit / FastAPI
```

---

مشروع أكاديمي لمادة نظم استرجاع المعلومات - كلية الهندسة المعلوماتية - جامعة دمشق.
