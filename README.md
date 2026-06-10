# 自然語言處理課程 RAG

這是一個本地 Ollama 版課程問答系統。它會把工作區內的 PDF 教材切成「投影片頁面」等級的 chunk，建立 metadata-aware hybrid retrieval，並用 Ollama 生成有來源引用的答案。

## 架構

```text
PDF pages
-> page-level chunks + metadata
-> BM25 tokens + Ollama embeddings
-> Weighted RRF hybrid retrieval
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

## 環境

可使用python -m venv venv創造虛擬環境

環境安裝完成後，安裝 Python 套件：

```powershell
python -m pip install -r requirements.txt
```

下載玩ollama後。確認 Ollama 已啟動，並下載模型：

```powershell
ollama pull bge-m3
ollama pull qwen2.5:7b
```

## 建立索引

在本資料夾執行：

```powershell
python -m course_rag.ingest --data-dir "自然語言課程資訊" --reset
```

建立索引時一定會使用 Ollama embedding model，請先確認 Ollama 已啟動且已下載 `bge-m3`。

## CSV 批次答題

輸入 CSV 不包含標題列，每一列直接寫一個問題：

```csv
期末專題怎麼評分？
學習活動至少要完成幾次？
期末考的時間、地點和範圍是什麼？
```

```powershell
python -m course_rag.batch_answer --input questions.csv --output answers.csv
```

輸出不包含標題列與原始題目，每一列只會寫入答案。

如果需要另外輸出來源欄位，可加上：

```powershell
python -m course_rag.batch_answer --input questions.csv --output answers.csv --include-sources
```

此時每一列會依序寫入答案與來源，同樣不包含標題列。

## 目前設計重點

- 中文與英文混合 BM25 tokenization。
- BM25 與 embedding 各召回前 40 條，使用 Weighted RRF 合併排序。
- 對課務問題加權：課程介紹、學習活動、期末專題、Announcement 會被優先檢索。
- 每個答案都要求標示 `[S1]` 這種來源引用。
- 資料庫放在 `storage/course_rag.sqlite`。
