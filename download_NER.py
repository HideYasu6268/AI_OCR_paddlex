from transformers import AutoTokenizer, AutoModelForTokenClassification

# 保存先ディレクトリを指定
save_directory = r"C:\MyPython\AI_OCR_naver\models\stockmark-bert-base-japanese-char-ner"

# モデルとトークナイザーをダウンロード
tokenizer = AutoTokenizer.from_pretrained("stockmark/bert-base-japanese-char-ner")
model = AutoModelForTokenClassification.from_pretrained("stockmark/bert-base-japanese-char-ner")

# ローカルに保存
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"モデルとトークナイザーは {save_directory} に保存されました。")