# Information Retrieval System

نظام متكامل لاسترجاع المعلومات (Information Retrieval System) مبني وفق **Clean Architecture**، ويدعم الفهرسة الكاملة لمجموعتي **Quora** و **Webis-Touché 2020 v2**، وخمسة نماذج استرجاع، والتقييم باستخدام `qrels`، وواجهة Streamlit، وواجهة FastAPI.

> هذا الملف يشرح تشغيل المشروع من الصفر، تنزيل البيانات، بناء الفهارس، استكمال البناء بعد الانقطاع، تنفيذ البحث، التقييم، وإنشاء الرسومات.

---

## 1. المزايا الرئيسية (Main Features)

- معالجة مختلفة حسب طبيعة كل Dataset.
- قراءة الوثائق بأسلوب Streaming من دون تحميل المجموعة كاملة في RAM.
- بناء الفهارس على دفعات (Batch Processing).
- فهرس نصي كامل على SQLite لخوارزميتي BM25 وTF-IDF.
- فهرس دلالي كامل باستخدام Sentence Transformers وFAISS.
- Checkpoint وResume أثناء بناء فهرس Embedding.
- خمسة نماذج استرجاع:
  - TF-IDF
  - BM25
  - Embedding
  - Hybrid Serial
  - Hybrid Parallel
- Query Refinement اختياري.
- Document Clustering باستخدام MiniBatchKMeans.
- تقييم باستخدام MAP@K وRecall@K وPrecision@10 وnDCG@K.
- قياس زمن الاستعلام (Latency Benchmarking).
- إنشاء Charts جاهزة للتقرير.
- واجهة Streamlit وREST API باستخدام FastAPI.

---

## 2. مجموعات البيانات الرسمية (Official Datasets)

| Dataset | الاسم داخل الأوامر | المهمة | عدد الوثائق | عدد الاستعلامات | عدد qrels |
|---|---|---|---:|---:|---:|
| Quora | `quora` | Duplicate-question retrieval | 522,931 | 10,000 | 15,675 |
| Webis-Touché 2020 v2 | `touche2020-v2` | Argument retrieval | 382,545 | 49 | 2,214 |

- معالجة Quora تستخدم الملف `question` لأنها تحتوي على أسئلة قصيرة.
- معالجة Touché تستخدم الملف `argument` لأنها تحتوي على حجج أطول مع `title` و`stance` و`url`.
- Natural Questions موجودة كإعداد قديم (legacy) وليست من مجموعات التسليم النهائية.

---

## 3. نماذج الاسترجاع (Retrieval Models)

- **TF-IDF:** تمثيل الكلمات بأوزان TF-IDF واستخدام Cosine Similarity.
- **BM25:** نموذج معجمي يدعم تطبيع طول الوثيقة ومعاملَي `k1` و`b`.
- **Embedding:** تمثيل دلالي باستخدام:

```text
sentence-transformers/all-MiniLM-L6-v2
```

- **Hybrid Serial:** استرجاع مرشحين باستخدام BM25، ثم إعادة ترتيبهم دلالياً باستخدام Embedding.
- **Hybrid Parallel:** تنفيذ TF-IDF وBM25 وEmbedding، ثم دمج الترتيب باستخدام Reciprocal Rank Fusion (RRF).

---

## 4. بنية المشروع (Project Architecture)

```text
app/
  domain/          النماذج والكيانات الأساسية
  application/     الخدمات وحالات الاستخدام وتنسيق العمليات
  infrastructure/  البيانات، المعالجة، الفهارس، الخوارزميات، التقييم
  presentation/    Streamlit وFastAPI
  shared/          الأدوات والثوابت المشتركة
scripts/           أوامر الفهرسة والبحث والتقييم والرسومات
tests/             الاختبارات الآلية
storage/           الفهارس والنتائج المحلية
docs/              التوثيق
```

قاعدة مختصرة لفهم Clean Architecture:

```text
Domain يعرّف
Application ينسّق
Infrastructure ينفّذ
Presentation يعرض
```

---

# التشغيل من الصفر (Complete Setup From Scratch)

## 5. المتطلبات (Requirements)

تم اختبار المشروع على:

- Windows 10/11
- Python 3.13
- RAM: 8 GB
- SSD
- CPU: Intel Core i5-1135G7
- لا تحتاج إلى GPU منفصل

المساحة المقترحة:

- الحد الأدنى العملي: 10 GB فارغة.
- الأفضل: 15–20 GB فارغة، لأن البيانات والفهارس والملفات المؤقتة قد تتراكم.

> عند بناء الفهارس الطويلة، يفضّل توصيل اللابتوب بالشاحن وتعطيل Sleep مؤقتاً.

---

## 6. تنزيل المشروع وإنشاء البيئة

```bash
git clone https://github.com/Hasan-Alhendi/information-retrieval-system.git
cd information-retrieval-system
python -m venv .env
```

تفعيل البيئة في Command Prompt:

```bat
.env\Scripts\activate
```

أو في PowerShell:

```powershell
.env\Scripts\Activate.ps1
```

تثبيت المكتبات:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### ملاحظة زمنية

قد يستغرق تثبيت المكتبات وتنزيل النماذج **10–30 دقيقة أو أكثر** حسب سرعة الإنترنت. أول استخدام لـSentence Transformer قد ينزّل النموذج من Hugging Face.

ظهور التحذير التالي لا يمنع التشغيل:

```text
You are sending unauthenticated requests to the HF Hub
```

يمكن استخدام `HF_TOKEN` اختيارياً لرفع حدود التنزيل، لكنه ليس مطلوباً لتشغيل المشروع.

---

## 7. تشغيل الاختبارات أولاً

```bash
pytest
```

النتيجة المتوقعة في النسخة الحالية:

```text
29 passed
```

إذا فشل اختبار، لا تبدأ الفهرسة الكاملة قبل إصلاحه.

---

## 8. التحقق من Dataset قبل الفهرسة

```bash
python scripts/validate_full_dataset.py --dataset quora
python scripts/validate_full_dataset.py --dataset touche2020-v2
```

يتحقق الأمر من:

- عدد الوثائق والاستعلامات وqrels.
- وجود qrels موجبة.
- شكل الوثائق.
- تطابق معرفات الاستعلامات.
- المساحة الحرة على القرص.

في التشغيل الأول قد يتم تنزيل ملفات Dataset تلقائياً.

### ملاحظة زمنية

التنزيل والتحقق قد يستغرقان **عدة دقائق إلى نصف ساعة** حسب سرعة الإنترنت وحالة Cache.

---

# الاختبار المصغر قبل الفهرسة الكاملة

## 9. Smoke Test للفهرس النصي

Quora:

```bash
python scripts/smoke_test_full_index.py --dataset quora
```

Touché:

```bash
python scripts/smoke_test_full_index.py --dataset touche2020-v2
```

يبني هذا الاختبار فهرساً صغيراً على 5,000 وثيقة تقريباً، ثم يجرّب BM25 وTF-IDF.

### الزمن المتوقع

عادة من **دقيقة إلى عدة دقائق** حسب Dataset والجهاز.

---

## 10. Smoke Test لفهرس Embedding

Quora:

```bash
python scripts/smoke_test_full_embedding.py --dataset quora --checkpoint-size 1000 --force
```

Touché:

```bash
python scripts/smoke_test_full_embedding.py --dataset touche2020-v2 --checkpoint-size 1000 --force
```

> استخدام `--force` هنا آمن لأن هذا Smoke Test منفصل ومؤقت. لا تستخدم `--force` مع الفهرس الكامل إلا إذا كنت تريد حذفه وإعادة بنائه من البداية.

### الزمن المتوقع

عادة من **2 إلى 10 دقائق**، وقد يزيد في أول مرة بسبب تنزيل وتحميل نموذج Embedding.

---

# بناء الفهارس الكاملة (Full-Corpus Indexing)

## 11. قواعد مهمة قبل البدء

1. الفهرس النصي الواحد يخدم BM25 وTF-IDF معاً.
2. فهرس Embedding مستقل ويستخدم FAISS.
3. يفضّل بناء كل Component وحده.
4. لا تغلق Terminal أثناء الكتابة النهائية للفهرس.
5. عند الانقطاع أعد الأمر نفسه من دون `--force` ليستكمل من آخر Checkpoint.
6. `--force` يحذف الفهرس المختار ويعيد البناء من الصفر.

أماكن الملفات:

```text
storage/indexes/<dataset>/disk_lexical/full/lexical.sqlite3
storage/vector_stores/<dataset>/embedding_sentence-transformers__all-MiniLM-L6-v2/full/
```

---

## 12. بناء Quora كاملة

### أ. الفهرس النصي BM25 + TF-IDF

```bash
python scripts/build_full_dataset.py --dataset quora --components lexical
```

### ب. فهرس Embedding

```bash
python scripts/build_full_dataset.py --dataset quora --components embedding
```

### أزمنة تقريبية على جهاز CPU مع 8 GB RAM وSSD

| العملية | الزمن التقريبي |
|---|---:|
| Quora Lexical Index | 30–90 دقيقة، وقد يزيد |
| Quora Embedding Index | 35–90 دقيقة |

هذه أرقام تقريبية وليست ثابتة. ضغط الجهاز، سرعة SSD، سرعة المعالج، ووجود Cache تؤثر في الزمن.

---

## 13. بناء Touché كاملة

### أ. الفهرس النصي BM25 + TF-IDF

```bash
python scripts/build_full_dataset.py --dataset touche2020-v2 --components lexical
```

### ب. فهرس Embedding

```bash
python scripts/build_full_dataset.py --dataset touche2020-v2 --components embedding
```

### أزمنة تقريبية

| العملية | الزمن التقريبي |
|---|---:|
| Touché Lexical Index | من ساعة إلى 3 ساعات، وقد يتجاوز ذلك |
| Touché Embedding Index | 45 دقيقة إلى ساعتين |

> نصوص Touché أطول من Quora، لذلك قد يستغرق الفهرس وقتاً أكبر رغم أن عدد وثائقها أقل.

---

## 14. بناء المكوّنين بأمر واحد — اختياري

```bash
python scripts/build_full_dataset.py --dataset quora --components lexical embedding
python scripts/build_full_dataset.py --dataset touche2020-v2 --components lexical embedding
```

لا ننصح بهذا على جهاز محدود الموارد؛ الأفضل تشغيل Lexical ثم Embedding بشكل منفصل لتسهيل المتابعة والتعافي من الأخطاء.

---

## 15. متابعة البناء بعد الانقطاع (Resume)

أعد الأمر نفسه فقط:

```bash
python scripts/build_full_dataset.py --dataset quora --components embedding
```

أو:

```bash
python scripts/build_full_dataset.py --dataset touche2020-v2 --components lexical
```

لا تضف `--force`.

يعتمد النظام على:

- `processed_documents`
- Checkpoints
- Metadata
- تطابق عدد متجهات FAISS مع عدد السجلات

لاستكمال البناء من آخر نقطة محفوظة.

---

## 16. فحص حالة الفهارس

```bash
python scripts/full_dataset_status.py --dataset quora
python scripts/full_dataset_status.py --dataset touche2020-v2
```

الحالة النهائية المطلوبة:

```text
lexical.finalized = true
embedding.finalized = true
ready_for_all_models = true
```

---

## 17. معالجة خطأ SQLite أثناء Finalization

قد يظهر في بناء فهرس نصي كبير:

```text
sqlite3.DatabaseError: database disk image is malformed
```

لا تحذف الفهرس ولا تستخدم `--force` مباشرة.

التشخيص:

```bash
python scripts/repair_lexical_index.py --dataset quora --diagnose-only
```

الإصلاح والاستكمال:

```bash
python scripts/repair_lexical_index.py --dataset quora --batch-size 256
```

ولـTouché:

```bash
python scripts/repair_lexical_index.py --dataset touche2020-v2 --batch-size 128
```

تأخذ أداة الإصلاح نسخة احتياطية عند الحاجة، وتعيد بناء الفهرس الثانوي المشتق، ثم تستكمل TF-IDF Norms من دون إعادة فهرسة كل الوثائق.

---

# البحث بعد اكتمال الفهارس

## 18. BM25 وTF-IDF

```bash
python scripts/search_full.py --dataset quora --model bm25 --query "How can I learn programming?" --top-k 10
python scripts/search_full.py --dataset quora --model tfidf --query "How can I learn programming?" --top-k 10
```

```bash
python scripts/search_full.py --dataset touche2020-v2 --model bm25 --query "Should teachers get tenure?" --top-k 10
python scripts/search_full.py --dataset touche2020-v2 --model tfidf --query "Should teachers get tenure?" --top-k 10
```

---

## 19. Embedding

```bash
python scripts/search_full_embedding.py --dataset quora --query "How can I learn programming?" --top-k 10 --batch-size 32
```

```bash
python scripts/search_full_embedding.py --dataset touche2020-v2 --query "Should teachers get tenure?" --top-k 10 --batch-size 16
```

### ملاحظة عن الزمن

أول استعلام داخل عملية Python جديدة قد يستغرق **8–15 ثانية** لأنه يحمل نموذج Sentence Transformer وفهرس FAISS. بعد Warm-up تصبح الاستعلامات اللاحقة أسرع بكثير، غالباً ضمن عشرات أو مئات milliseconds حسب النموذج والبيانات.

---

## 20. Hybrid Serial وHybrid Parallel

```bash
python scripts/search_full_hybrid.py --dataset quora --model hybrid_serial --query "How can I learn programming?" --top-k 10
python scripts/search_full_hybrid.py --dataset quora --model hybrid_parallel --query "How can I learn programming?" --top-k 10
```

```bash
python scripts/search_full_hybrid.py --dataset touche2020-v2 --model hybrid_serial --query "Should teachers get tenure?" --top-k 10
python scripts/search_full_hybrid.py --dataset touche2020-v2 --model hybrid_parallel --query "Should teachers get tenure?" --top-k 10
```

Hybrid Parallel أبطأ عادة لأنه يشغّل ثلاثة نماذج ثم يدمج نتائجها.

---

# تشغيل الواجهات

## 21. واجهة Streamlit

```bash
streamlit run app/presentation/streamlit/ui.py
```

افتح:

```text
http://localhost:8501
```

الواجهة تحتوي على:

- Search
- Evaluation
- Datasets
- Clustering
- اختيار Full Dataset أو Development Subset
- النماذج الخمسة
- Query Refinement
- معاملات BM25

> يجب بناء فهارس Full Dataset قبل اختيار Full Dataset من الواجهة.

---

## 22. واجهة FastAPI

```bash
uvicorn app.main:app --reload
```

افتح:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

مثال طلب بحث كامل:

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

أرسله إلى:

```text
POST /search
```

القيمة:

```json
"max_docs": null
```

تعني استخدام الفهرس الكامل.

---

# التقييم والرسومات

## 23. تقييم سريع للتأكد من التشغيل

Quora على 100 استعلام:

```bash
python scripts/evaluate_full_system.py --dataset quora --max-queries 100 --top-k 10
```

Touché على 10 استعلامات:

```bash
python scripts/evaluate_full_system.py --dataset touche2020-v2 --max-queries 10 --top-k 10
```

### الزمن المتوقع

- Quora / 100 queries: عدة دقائق.
- Touché / 10 queries: عدة دقائق.

---

## 24. التقييم الرسمي الكامل

Quora على جميع 10,000 استعلام:

```bash
python scripts/evaluate_full_system.py --dataset quora --max-queries 10000 --top-k 10
```

Touché على جميع 49 استعلاماً:

```bash
python scripts/evaluate_full_system.py --dataset touche2020-v2 --max-queries 49 --top-k 10
```

### تنبيه زمني مهم

| العملية | الزمن التقريبي |
|---|---:|
| Quora، خمسة نماذج، 10,000 Query | من ساعة إلى 4 ساعات، وقد يزيد |
| Touché، خمسة نماذج، 49 Query | عدة دقائق إلى نحو 20 دقيقة |

لا توقف تقييم Quora الكامل إذا استمر لساعات؛ هذا طبيعي لأنه ينفذ خمسة نماذج على 10,000 استعلام.

النتائج تحفظ في:

```text
storage/evaluation/<dataset>_full_system_evaluation.csv
```

---

## 25. Benchmark لزمن BM25 وTF-IDF

```bash
python scripts/benchmark_full_search.py --dataset quora --models bm25 tfidf --max-queries 10 --repeats 3 --warmups 1 --top-k 10
```

```bash
python scripts/benchmark_full_search.py --dataset touche2020-v2 --models bm25 tfidf --max-queries 10 --repeats 3 --warmups 1 --top-k 10
```

---

## 26. Benchmark لـEmbedding وHybrid

```bash
python scripts/benchmark_dense_hybrids.py --dataset quora --models embedding hybrid_serial hybrid_parallel --max-queries 10 --repeats 3 --top-k 10
```

```bash
python scripts/benchmark_dense_hybrids.py --dataset touche2020-v2 --models embedding hybrid_serial hybrid_parallel --max-queries 10 --repeats 3 --top-k 10
```

يستخدم Benchmark عملية Python واحدة وWarm-up، لذلك لا يخلط زمن تحميل النموذج مع زمن الاستعلام الحقيقي.

---

## 27. إنشاء الرسومات

```bash
python scripts/generate_evaluation_charts.py --dataset quora
python scripts/generate_evaluation_charts.py --dataset touche2020-v2
```

تحفظ الرسومات في:

```text
storage/evaluation/charts/quora/
storage/evaluation/charts/touche2020-v2/
```

الرسومات قد تستبدل ملفات قديمة تحمل الاسم نفسه داخل مجلد Dataset نفسه.

---

# النتائج المرجعية النهائية

## 28. Quora — 522,931 وثيقة و10,000 استعلام

| Model | MAP@10 | Recall@10 | Precision@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.6565 | 0.8116 | 0.1086 | 0.7073 |
| BM25 | 0.6898 | 0.8434 | 0.1135 | 0.7399 |
| Embedding | **0.8363** | **0.9503** | **0.1337** | **0.8755** |
| Hybrid Serial | 0.8153 | 0.9279 | 0.1291 | 0.8548 |
| Hybrid Parallel | 0.7549 | 0.8875 | 0.1214 | 0.8003 |

أفضل نموذج في Quora هو Embedding لأن المهمة تعتمد على التشابه الدلالي بين الأسئلة المعاد صياغتها.

---

## 29. Touché — 382,545 وثيقة و49 استعلاماً

| Model | MAP@10 | Recall@10 | Precision@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.0563 | 0.1183 | 0.1735 | 0.1591 |
| BM25 | **0.1121** | 0.1892 | 0.2673 | **0.2861** |
| Embedding | 0.0758 | 0.1443 | 0.1857 | 0.1886 |
| Hybrid Serial | 0.1112 | **0.1902** | **0.2714** | 0.2826 |
| Hybrid Parallel | 0.0994 | 0.1742 | 0.2510 | 0.2531 |

حقق BM25 أفضل MAP@10 وnDCG@10، بينما حقق Hybrid Serial أفضل Recall@10 وPrecision@10.

---

# ميزات إضافية

## 30. Query Refinement

الميزة اختيارية من الواجهة أو API، وتتضمن:

- Normalization
- قاموس تصحيح أخطاء محدود Rule-based
- Query Expansion خاص بمصطلحات IR

مثال:

```text
infomation retrival querry
```

يصبح تقريباً:

```text
information retrieval query search ranking question
```

هذه ليست أداة تصحيح إنجليزية عامة، بل استراتيجية محدودة وقابلة للتفسير.

---

## 31. Document Clustering

من تبويب Clustering في Streamlit:

- اختيار Dataset.
- تحديد عدد Clusters.
- تحديد عدد وثائق Development Subset.
- استخدام Embeddings.
- تشغيل MiniBatchKMeans.
- عرض حجم كل Cluster وTop Terms وأمثلة وثائق.

Clustering مخصصة للتجربة على Development Subset ولا تحتاج إلى إعادة بناء فهارس البحث الكاملة.

---

# حل المشكلات (Troubleshooting)

## 32. الفهرس غير جاهز

إذا ظهر:

```text
Full index is not ready
```

نفّذ:

```bash
python scripts/full_dataset_status.py --dataset <dataset-name>
```

وابنِ Component الناقص.

---

## 33. توقف بناء Embedding

أعد الأمر نفسه من دون `--force`:

```bash
python scripts/build_full_dataset.py --dataset <dataset-name> --components embedding
```

سيكمل من آخر Checkpoint.

---

## 34. بطء أول استعلام

هذا يسمى Cold Start، وينتج عن تحميل:

- Sentence Transformer
- FAISS index
- SQLite metadata

استخدم Streamlit أو FastAPI، لأنهما يحتفظان بالنماذج والفهارس في الذاكرة بين الاستعلامات.

---

## 35. نفاد RAM

- لا تحمّل Dataset كاملة في Python List.
- استخدم أوامر Full Dataset المبنية على Streaming.
- قلل `embedding-batch-size` عند الحاجة:

```bash
python scripts/build_full_dataset.py --dataset quora --components embedding --embedding-batch-size 16
```

- لا تشغّل Build وEvaluation وStreamlit في الوقت نفسه على جهاز 8 GB RAM.

---

## 36. نفاد مساحة القرص

احذف فقط الملفات المؤقتة أو Smoke Tests التي لم تعد مطلوبة. لا تحذف مجلدات `full` إلا إذا كنت تريد إعادة الفهرسة.

تحقق من الحالة والحجم:

```bash
python scripts/full_dataset_status.py --dataset quora
python scripts/full_dataset_status.py --dataset touche2020-v2
```

---

# تسلسل التشغيل المختصر

```text
1. git clone
2. إنشاء virtual environment
3. pip install -r requirements.txt
4. pytest
5. validate_full_dataset
6. smoke_test_full_index
7. smoke_test_full_embedding
8. build_full_dataset --components lexical
9. build_full_dataset --components embedding
10. full_dataset_status
11. search_full / search_full_embedding / search_full_hybrid
12. evaluate_full_system
13. benchmark scripts
14. generate_evaluation_charts
15. Streamlit أو FastAPI
```

---

## الترخيص والاستخدام

هذا المشروع أكاديمي لمادة نظم استرجاع المعلومات، كلية الهندسة المعلوماتية، جامعة دمشق.