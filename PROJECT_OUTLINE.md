# 自然語言處理課程 RAG 專案大綱

## 一、專案目標

- 建立本地課程問答系統。
- 使用課程 PDF 作為知識來源。
- 根據檢索到的教材內容回答問題。
- 回答附帶來源頁碼，降低模型幻覺。

## 二、使用模型與工具

- Embedding model：`bge-m3`
- Chat model：`qwen2.5:7b`
- 模型執行平台：Ollama
- PDF 文字擷取：pypdf
- 資料儲存：SQLite

## 三、建立索引流程

執行指令：

```powershell
python -m course_rag.ingest --data-dir "自然語言課程資訊" --reset
```

索引建立流程：

```text
讀取資料夾內的 PDF
→ 使用 pypdf 擷取每頁文字
→ 每個 PDF 頁面建立一個 chunk
→ 判斷文件與 chunk 類別
→ 產生 BM25 tokens
→ 使用 bge-m3 產生 embedding
→ 儲存至 SQLite
```

每個 chunk 儲存內容：

```text
PDF 檔名
文件 ID
頁碼
標題
Category metadata
頁面文字
BM25 tokens
Embedding vector
```

## 四、Chunk 類別判斷

根據檔名分類：

```text
期末專題 PDF        → project
學習活動 PDF        → activity
課程介紹 PDF        → course_info
一般課程教材        → lecture
```

一般教材頁面若包含期中考、期末考、地點、評分等行政資訊，該頁會分類為：

```text
announcement
```

## 五、使用者問題分類

系統使用關鍵字判斷問題類型：

```text
專題、報告、測資、分組 → project
學習活動、練習、問卷   → activity
期中、期末、地點、時間 → admin
其他問題               → concept
```

分類結果會用於調整檢索結果的重要度。

## 六、檢索流程

```text
使用者問題
→ BM25 關鍵字檢索
→ bge-m3 語意檢索
→ Weighted RRF 合併排序
→ Metadata 與特殊規則加權
→ 取前 8 個結果
→ 加入命中頁面的前後頁
→ 最多保留 12 個來源頁面
```

### BM25 檢索

- 對全部 chunks 計算關鍵字相關分數。
- 取排名前 40 條。
- 適合日期、Email、專有名詞與精確文字。

### Embedding 檢索

- 使用 `bge-m3` 將問題轉換成向量。
- 與全部 chunk embeddings 計算 cosine similarity。
- 取排名前 40 條。
- 適合語意相近但用詞不同的問題。

### Weighted RRF

使用排名融合 BM25 與 Embedding：

```text
RRF score =
1.15 / (60 + BM25 rank)
+
1.00 / (60 + Embedding rank)
```

BM25 權重比 Embedding 高約 15%。

## 七、額外排序加權

RRF 分數計算完成後，還會加入：

- 問題類別與 chunk category 的匹配權重。
- 問題關鍵詞覆蓋率。
- 期中考與期末考特殊加權。
- 定義題特殊加權。

例如專題問題：

```text
project chunk      × 2.0
course_info chunk  × 1.2
announcement chunk × 1.1
lecture chunk      × 0.85
```

## 八、Context 建立

- 預設取得檢索排名前 8 個 chunks。
- 每個命中頁面加入前一頁與後一頁。
- 相同頁面不重複加入。
- 最多提供 12 個來源頁面。
- Context 總長度最多 12,000 字元。

## 九、答案生成流程

```text
使用者問題
+ 檢索來源內容
+ System Prompt
→ qwen2.5:7b
→ 繁體中文答案
→ 顯示來源引用
```

System Prompt 要求：

- 只能根據來源回答。
- 資料不足時明確說明。
- 使用繁體中文。
- 保留日期、比例、Email 等重要資訊。
- 不回答問題未詢問的內容。
- 重要句子標示 `[S1]` 等來源引用。

## 十、主要操作指令

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
