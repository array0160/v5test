# BookOCR Web v5 Experimental — Auto Layout Router

**v4.5 保留為穩定直書版。**
v5 是另外一條 Experimental 分支，不取代 v4.5。

## v5 的目標

一般使用者不用理解 OCR 類型。

首頁預設：

`Auto（推薦：直接亂丟）`

先跑 PP-OCRv5 detector，看文字區塊的形狀：

- 多數高瘦 + 圖片像跨頁 → 傳統直書
- 多數高瘦 + 單頁 → 直排單頁
- 多數寬扁 → 橫排書籍 / 文件
- 方向混合 → 一般圖片 / 招牌 / 海報

使用者仍然可以手動覆蓋：

- 傳統直書
- 橫排書籍 / 文件
- 一般圖片 / 招牌 / 海報

## 傳統直書

完整保留 v4.5：

UVDoc → Detector → PCA → V3 → A/B Recognition

並修正畫面閱讀方向：

- 右頁真的顯示在右邊
- 左頁顯示在左邊
- Column 01 真的在最右邊
- 全文仍是右頁 → 左頁、右欄 → 左欄

## 橫排書籍 / 文件

- 如果像跨頁：左頁 → 右頁
- 每頁 UVDoc
- Detector 找文字區塊
- 上 → 下
- 同一行左 → 右
- Recognition

## 直排單頁

不硬切左右頁。

Detector 找高瘦文字區塊，
依右 → 左排序，
RecognitionService 對高瘦 crop 自動旋正後辨識。

## 一般圖片 / 招牌

不跑書頁 V3。

原圖 → Detector → 區塊排序 → Recognition

這是第一版 router，之後可再針對：
- 斜招牌
- 多欄現代文件
- 表格
- 多語言
做更細的 layout 判斷。

## GitHub

你現有 workflow 不用改。

Code → Add file → Upload files → 覆蓋新版 → Commit → Actions 綠色 → Ctrl+F5
