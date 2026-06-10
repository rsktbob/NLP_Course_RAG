# 自然語言處理課程 RAG

這是一個本地課程問答系統。系統會將課程 PDF 依頁面建立索引，使用 BM25 與 Ollama embedding 召回候選頁面，再透過多語 reranker 重新排序，最後由 Ollama 根據檢索來源生成答案。

## 預設檢索流程

```text
使用者問題
-> BM25 Top 30
-> bge-m3 embedding Top 30
-> 合併並去除重複候選
-> BAAI/bge-reranker-v2-m3 重新排序
-> 取 Reranker Top 3
-> 建立 Context
-> qwen2.5:7b 生成答案
```

目前不使用 RRF，也不依照問題類別或 metadata 人工加權。`category` 會儲存在 chunk 中並提供給模型參考，但候選頁面的最終重要度由 reranker 判斷。

## 預設模型

- Embedding：`bge-m3`，透過 Ollama 執行
- Reranker：`BAAI/bge-reranker-v2-m3`，透過 Hugging Face Transformers 執行
- Chat：`qwen2.5:7b`，透過 Ollama 執行

可以用環境變數覆蓋：

```powershell
$env:OLLAMA_EMBED_MODEL="bge-m3"
$env:RERANK_MODEL="BAAI/bge-reranker-v2-m3"
$env:OLLAMA_CHAT_MODEL="qwen2.5:7b"
$env:OLLAMA_HOST="http://localhost:11434"
```

## 安裝環境
建立虛擬環境
```powershell
python -m venv venv
```

建立並啟用虛擬環境後，安裝套件：

```powershell
python -m pip install -r requirements.txt
```

安裝並啟動 Ollama 後，下載模型：

```powershell
ollama pull bge-m3
ollama pull qwen2.5:7b
```

Reranker 會在第一次問答時由 Hugging Face 自動下載。

## 建立索引

```powershell
python -m course_rag.ingest --data-dir "自然語言課程資訊" --reset
```

建立索引時會：

1. 使用 `pypdf` 擷取每個 PDF 頁面的文字。
2. 將每個有文字的頁面建立為一個 chunk。
3. 為 chunk 加入檔名、頁碼、標題與 category metadata。
4. 建立中英文混合 BM25 tokens。
5. 使用 `bge-m3` 產生 embedding。
6. 將索引儲存至 `storage/course_rag.sqlite`。

建立索引一定會使用 embedding，請先確認 Ollama 已啟動且已下載 `bge-m3`。

## CSV 批次答題

輸入 CSV 不包含標題列，每一列直接寫一個問題：

```csv
期末專題怎麼評分？
學習活動至少要完成幾次？
期末考的時間、地點和範圍是什麼？
```

執行批次答題：

```powershell
python -m course_rag.batch_answer --input eval_questions.csv --output eval_answers.csv
```

輸出同樣不包含標題列，每一列依序為：

```text
原始題目,答案
```

加入來源欄位：

```powershell
python -m course_rag.batch_answer --input eval_questions.csv --output eval_answers.csv --include-sources
```

此時每一列依序為：

```text
原始題目,答案,來源
```


## 常用檢索參數

```powershell
python -m course_rag.ask "問題" --candidate-k 30 --top-k 3 --context-window 0
```

- `--candidate-k`：BM25 與 embedding 各自召回的候選數量，預設為 `30`。
- `--top-k`：reranker 排序後保留的 chunk 數量，預設為 `3`。
- `--context-window`：每個命中頁面額外加入的前後頁數量，預設為 `0`。

Context 最多包含 12 個來源頁面，總長度最多約 12,000 字元。
