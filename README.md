# 自然語言處理課程 RAG

這是一個本地 Ollama 版課程問答系統。它會把工作區內的 PDF 教材切成「投影片頁面」等級的 chunk，建立 metadata-aware hybrid retrieval，並用 Ollama 生成有來源引用的答案。

## 架構

```text
PDF pages
-> page-level chunks + metadata
-> BM25 tokens + Ollama embeddings
-> hybrid retrieval
-> neighbor page context
-> Ollama chat answer with citations
```

預設模型：

- Embedding: `bge-m3`
- Chat: `qwen2.5:7b`

可以用環境變數覆蓋：

```powershell
$env:OLLAMA_EMBED_MODEL="bge-m3"
$env:OLLAMA_CHAT_MODEL="qwen2.5:7b"
$env:OLLAMA_HOST="http://localhost:11434"
```

## 安裝

先安裝 Python 套件：

```powershell
python -m pip install -r requirements.txt
```

如果 Windows 找不到 `python`，可以改用 `py`：

```powershell
py -m pip install -r requirements.txt
```

確認 Ollama 已啟動，並下載模型：

```powershell
ollama pull bge-m3
ollama pull qwen3.5:9b
```

## 建立索引

在本資料夾執行：

```powershell
python -m course_rag.ingest --data-dir . --reset
```

如果教材 PDF 放在子資料夾，例如目前的 `自然語言課程資訊`，可以明確指定：

```powershell
python -m course_rag.ingest --data-dir "自然語言課程資訊" --reset
```

如果還沒安裝 Ollama 或 embedding model，可以先建立 BM25-only 測試索引：

```powershell
python -m course_rag.ingest --data-dir . --reset --no-embeddings
```

目前不用 Ollama 的測試索引建議用：

```powershell
python -m course_rag.ingest --data-dir "自然語言課程資訊" --reset --no-embeddings
```

## 單題問答

```powershell
python -m course_rag.ask "期末專題怎麼評分？" --show-sources
```

沒有 Ollama 時可以先測檢索結果：

```powershell
python -m course_rag.ask "期末專題怎麼評分？" --no-vector --no-llm
```

互動模式：

```powershell
python -m course_rag.ask --show-sources
```

## CSV 批次答題

輸入 CSV 可以有 `題目` 欄位；如果沒有 header，系統會使用第一欄當題目。

```powershell
python -m course_rag.batch_answer --input questions.csv --output answers.csv
```

輸出會包含：

- `題目`
- `答案`

如果需要另外輸出來源欄位，可加上：

```powershell
python -m course_rag.batch_answer --input questions.csv --output answers.csv --include-sources
```

## 目前設計重點

- 中文與英文混合 BM25 tokenization。
- 對課務問題加權：課程介紹、學習活動、期末專題、Announcement 會被優先檢索。
- 每個答案都要求標示 `[S1]` 這種來源引用。
- 資料庫放在 `storage/course_rag.sqlite`。
