from paddleocr import PaddleOCR

print("PaddleOCRモデルのダウンロードを開始します...")

# モデルの初期化（自動ダウンロード）
ocr = PaddleOCR(
    lang='japan',  # 日本語を含む多言語
    use_textline_orientation=True
)

print("モデルのダウンロードが完了しました！")
print("モデルは以下のディレクトリに保存されています:")
print("C:\\Users\\yasud\\.paddlex\\official_models")
