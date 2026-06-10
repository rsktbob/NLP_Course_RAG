# 自然語言處理課程 RAG 專案大綱

## 一、專案目標

- 建立以課程 PDF 為知識來源的本地問答系統。
- 結合關鍵字與語意檢索，提高不同問題形式的召回率。
- 使用 reranker 從候選頁面中找出最相關內容。
- 限制模型只能根據檢索來源回答，並提供來源頁碼。

## 二、使用模型與工具

| 功能 | 模型或工具 | 執行方式 |
| --- | --- | --- |
| PDF 文字擷取 | `pypdf` | Python |
| BM25 | 專案內建 BM25 | Python |
| Embedding | `bge-m3` | Ollama |
| Reranker | `BAAI/bge-reranker-v2-m3` | Hugging Face Transformers |
| 答案生成 | `qwen2.5:7b` | Ollama |
| 索引儲存 | SQLite | `storage/course_rag.sqlite` |

Reranker 會自動偵測 CUDA；若 PyTorch 無法使用 CUDA，則回退至 CPU。

## 三、建立索引流程

執行指令：

```powershell
python -m course_rag.ingest --data-dir "自然語言課程資訊" --reset
```

索引建立流程：

```text
讀取資料夾內的 PDF
-> 使用 pypdf 擷取每頁文字
-> 每個有文字的 PDF 頁面建立一個 chunk
-> 判斷文件與 chunk category
-> 產生中英文混合 BM25 tokens
-> 使用 bge-m3 產生 embedding
-> 儲存至 SQLite
```

每個 chunk 儲存：

- PDF 檔名與文件 ID
- 頁碼範圍
- 頁面標題
- Category metadata
- 頁面文字
- BM25 tokens
- Embedding vector

系統沒有停用 embedding 的模式；建立索引與問答檢索都會使用 `bge-m3`。

## 四、Category Metadata

文件會先根據檔名分類：

```text
期末專題 PDF       -> project
學習活動 PDF       -> activity
課程介紹或 c0_ PDF -> course_info
一般課程教材       -> lecture
```

一般教材頁面若包含考試、地點、評分、繳交等行政資訊，該 chunk 會分類為 `announcement`。

Category 會：

- 儲存在 SQLite 的 chunk metadata。
- 加入 BM25 token 內容。
- 放入提供給 LLM 的來源 Context。

目前不會根據使用者問題進行人工類別判斷，也不會對特定 category 額外加權。

## 五、檢索流程

```text
使用者問題
-> BM25 召回 Top 30
-> bge-m3 embedding 召回 Top 30
-> 合併並去除重複 chunk
-> bge-reranker-v2-m3 對候選重新評分
-> 保留 Reranker Top 3
-> 建立答案 Context
```

### BM25 檢索

- 將問題切成中英文混合 tokens。
- 對所有 chunks 計算 BM25 分數。
- 預設取前 30 條。
- 適合日期、Email、專有名詞與精確文字。

### Embedding 檢索

- 使用 `bge-m3` 將問題轉換成向量。
- 與所有 chunk embeddings 計算 cosine similarity。
- 預設取前 30 條。
- 適合語意相近但用詞不同的問題。

### 合併候選

- 將 BM25 Top 30 與 Embedding Top 30 合併。
- 以 chunk ID 去除重複結果。
- 去重前最多 60 條，去重後通常少於 60 條。

### Reranker

- 同時閱讀問題與每個候選頁面的檔名、標題及文字。
- 為每個候選計算相關分數並重新排序。
- 預設只保留分數最高的 3 個 chunks。
- 目前不使用 RRF，也不使用 BM25/embedding 權重加總或人工 metadata 權重。

## 六、Context 建立

- 預設使用 reranker 排名前 3 的 chunks。
- `context_window=0`，預設不加入命中頁面的前後頁。
- 相同文件的相同頁面不會重複加入。
- Context 最多包含 12 個來源頁面。
- Context 總長度最多約 12,000 字元。

## 七、答案生成流程

```text
使用者問題
+ Reranker 選出的來源 Context
+ System Prompt
-> qwen2.5:7b
-> 繁體中文答案與來源引用
```

System Prompt 要求：

- 只能根據提供的來源回答。
- 資料不足時明確說明。
- 預設使用繁體中文。
- 保留日期、比例、地點、Email 等重要資訊。
- 只回答問題詢問的面向。
- 答案控制在 1 至 5 句或 3 至 5 個條列。
- 重要句子標示 `[S1]` 等來源引用。

## 八、輸出格式

單題問答直接輸出答案，不加入「題目」或「答案」標籤。

批次輸入 CSV 不包含標題列，每列直接放一個問題。批次輸出也不包含標題列：

```text
預設：原始題目,答案
使用 --include-sources：原始題目,答案,來源
```

批次答案預設移除 `[S1]` 等引用標記；使用 `--keep-citations` 可保留。

## 九、主要操作指令

一般套件安裝：

```powershell
python -m pip install -r requirements.txt
```

CUDA 套件安裝：

```powershell
python -m pip install -r requirements-cuda.txt
```

建立索引：

```powershell
python -m course_rag.ingest --data-dir "自然語言課程資訊" --reset
```

單題問答：

```powershell
python -m course_rag.ask "期末專題怎麼評分？" --show-sources
```

互動問答：

```powershell
python -m course_rag.ask --show-sources
```

批次答題：

```powershell
python -m course_rag.batch_answer --input eval_questions.csv --output eval_answers.csv --include-sources
```
