# Interactive SVM Kernel Trick 3D Demo

A Streamlit app that demonstrates the SVM kernel trick on concentric-ring data:
adjust the kernel and hyperparameters in the sidebar and watch the 2D decision
boundary and 3D decision-function surface update live.

**Live demo:** https://wi-svm-hw.streamlit.app/

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) -> **New app** -> select the repo, branch, and `app.py`.
3. Click **Deploy**. Future pushes to that branch auto-redeploy.

## Workflow (開發流程)

這個 repo 是用 [Claude Code](https://claude.com/claude-code) 互動式開發出來的。大致流程:

1. **資料與模型**: 用 `sklearn.datasets.make_circles` 產生同心圓資料 (內圈/外圈
   各為一類, 在 2D 平面上線性不可分), 用 `sklearn.svm.SVC` 即時訓練, kernel
   (rbf/poly/linear)、C、gamma、degree、noise、點數、隨機種子皆為側邊欄可調參數。
2. **視覺化**: 用 Plotly 畫出 2D 決策邊界 (`f=0` 實線決策面 + `f=±1` 虛線 margin +
   support vectors 標記) 與 3D 決策函數曲面 `f(x, y)`, 兩者互動連動同一個模型結果。
3. **效能**: 用 `@st.cache_data` 快取資料生成、`@st.cache_resource` 快取模型訓練,
   避免每次互動都重新計算。
4. **視覺風格**: 自訂 `.streamlit/config.toml` 主題色彩, 注入 CSS 放大字體與標題,
   統一圖表配色與留白, 讓畫面更接近正式 dashboard 風格。
5. **驗證方式**: 沒有依賴單元測試, 而是直接啟動 `streamlit run app.py`, 用
   Puppeteer 驅動真實 Chrome 瀏覽器載入頁面、切換 kernel、截圖比對, 確保畫面真的
   如預期 (而不是只靠程式碼邏輯推測)。這個過程中抓到並修正了兩個實際發現的 bug:
   - `make_circles` 預設的內外圈標籤跟規格假設相反, 造成圖例顏色配對錯誤。
   - 字體放大後 "RBF (Gaussian)" 等較長的 kernel 名稱在指標卡片中被截斷。
6. **部署**: 推送到 GitHub repo, 再到 [Streamlit Community Cloud](https://share.streamlit.io)
   連結 repo 自動部署, 之後對該分支的 push 會自動觸發重新部署。
