import os
import re
import json
import traceback
from typing import List, Dict, Any

import torch
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class ReceiptParserUnified:
    """
    必要フィールドのみを返すレシートパーサ:
      - date (list[str])
      - company (list[str])  ← NERのorgとカスタムcompanyを統合
      - tax_rate (list[str])
      - total_amount (list[str])
      - registration_number (list[str])  ← 「T+13桁」限定
      - registration (bool) ← registration_number が1つ以上あれば True
    """

    def __init__(self):
        try:
            # --- OCR ---
            self.ocr = PaddleOCR(lang='japan')

            # --- NER ---
            model_path = r"C:\MyPython\AI_OCR_naver\models\xlm-roberta-ner-japanese"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)

            device = 0 if torch.cuda.is_available() else -1
            self.ner = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=device
            )
            print("✓ 初期化 OK（PaddleOCR + Transformers NER）")
        except Exception as e:
            print(f"✗ 初期化失敗: {e}")
            raise

    # =========================
    # 公開 API
    # =========================
    def parse_receipt(self, image_path: str) -> Dict[str, Any]:
        print("\n" + "=" * 50)
        print(f"領収書解析を開始: {image_path}")
        print("=" * 50)

        if not os.path.exists(image_path):
            msg = f"画像ファイルが見つかりません: {image_path}"
            print(f"エラー: {msg}")
            return {"error": msg, "raw_texts": []}

        try:
            texts = self._run_ocr(image_path)
            if not texts:
                return {"error": "テキストが抽出できませんでした", "raw_texts": []}

            # 統一スキーマ（必要なキーだけ）
            fields = self._empty_fields()

            # NER（Transformers）
            self._apply_transformers_ner(texts, fields)

            # カスタムルール
            self._apply_custom_rules(texts, fields)

            # 後処理（正規化・統合）
            self._normalize_fields(fields)

            # 結果整形
            result = self._format_result(fields, texts)
            return result

        except Exception as e:
            print(f"解析中にエラー: {e}")
            traceback.print_exc()
            return {"error": str(e), "raw_texts": []}

    # =========================
    # OCR
    # =========================
    def _run_ocr(self, image_path: str) -> List[str]:
        print("\n[ステップ1] OCR処理を実行中...")
        result = self.ocr.predict(image_path)
        texts = self._extract_texts(result)

        print(f"抽出されたテキスト ({len(texts)}件):")
        for i, t in enumerate(texts, 1):
            print(f"{i:3d}: {t}")
        return texts

    def _extract_texts(self, ocr_result: Any) -> List[str]:
        texts: List[str] = []
        try:
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                page = ocr_result[0]
                # 1) オブジェクト属性
                try:
                    rec_texts = page.rec_texts
                    for s in rec_texts:
                        s = str(s).strip()
                        if s:
                            texts.append(s)
                    return texts
                except AttributeError:
                    pass

                # 2) 辞書
                try:
                    if hasattr(page, '__getitem__'):
                        rec_texts = page['rec_texts']
                        for s in rec_texts:
                            s = str(s).strip()
                            if s:
                                texts.append(s)
                        return texts
                except Exception:
                    pass

                # 3) 旧来のlist形式
                if isinstance(page, list):
                    for line in page:
                        if line and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, (tuple, list)) and len(text_info) >= 1:
                                s = str(text_info[0]).strip()
                                if s:
                                    texts.append(s)
        except Exception as e:
            print(f"OCRテキスト抽出エラー: {e}")
            traceback.print_exc()

        return texts

    # =========================
    # 統一フィールド
    # =========================
    def _empty_fields(self) -> Dict[str, List[str]]:
        return {
            "date": [],
            "company": [],              # 最終的に list で返す
            "tax_rate": [],
            "total_amount": [],
            "registration_number": [],  # 「T+13桁」限定
            # 補助（内部計算用）
            "money": [],
            "org": [],
        }

    def _add(self, fields: Dict[str, List[str]], key: str, value: str):
        v = value.strip()
        if not v:
            return
        if v not in fields[key]:
            fields[key].append(v)

    # =========================
    # NER（Transformers）
    # =========================
    def _apply_transformers_ner(self, texts: List[str], fields: Dict[str, List[str]]):
        print("\n[ステップ2] NER（Transformers）...")
        try:
            cleaned = []
            for t in texts:
                t = re.sub(r'[　\s]+', ' ', t.strip())
                t = re.sub(r'[、。・]', ' ', t)
                if t:
                    cleaned.append(t)

            joined = " ".join(cleaned)[:3000]
            results = self.ner(joined)

            for ent in results:
                label = ent.get("entity_group", "")
                word = ent.get("word", "").strip()
                if not word or len(word) < 2:
                    continue

                if label in ("DATE",):
                    self._add(fields, "date", word)
                elif label in ("MONEY",):
                    self._add(fields, "money", word)
                elif label in ("ORG", "GPE", "LOC"):
                    self._add(fields, "org", word)

            # --- 補足（正規表現） ---
            for t in cleaned:
                # 金額
                for m in re.findall(r'[￥¥]\s*\d{1,3}(?:,\d{3})*(?:-\s*)?', t):
                    self._add(fields, "money", m.replace(" ", ""))

                # 日付
                for d in re.findall(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}', t):
                    self._add(fields, "date", d)

                # 税率
                for rate in re.findall(r'(?<!\d)(?:\d{1,2})[%％](?!\d)', t):
                    self._add(fields, "tax_rate", rate.replace('％', '%'))

                # T13桁
                for regnum in re.findall(r'T\d{13}', t):
                    self._add(fields, "registration_number", regnum)

        except Exception as e:
            print(f"NER処理中にエラー: {e}")

    # =========================
    # カスタムルール
    # =========================
    def _apply_custom_rules(self, texts: List[str], fields: Dict[str, List[str]]):
        print("- カスタムルールを適用...")
        for s in texts:
            s = s.strip()
            if not s:
                continue

            # 日付
            for pat in [r'\d{4}/\d{1,2}/\d{1,2}', r'\d{4}年\d{1,2}月\d{1,2}日',
                        r'\d{2}/\d{1,2}/\d{1,2}', r'令和\d+年\d+月\d+日']:
                for m in re.findall(pat, s):
                    self._add(fields, "date", m)

            # 金額（候補）
            for pat in [r'￥\s*[\d,]+-?', r'¥\s*[\d,]+-?']:
                for m in re.findall(pat, s):
                    self._add(fields, "money", m.replace(" ", ""))

            # 会社・店舗候補
            for pat in [r'.+店', r'.+株式会社', r'.+有限会社', r'.+Co\.', r'.+Ltd\.', r'.+商店']:
                m = re.findall(pat, s)
                for v in m:
                    self._add(fields, "company", v.strip('「」[]<>'))

            # T13桁（インボイス登録番号限定）
            for regnum in re.findall(r'T\d{13}', s):
                self._add(fields, "registration_number", regnum)

            # 税率
            for m in re.findall(r'\d{1,2}[%％]', s):
                self._add(fields, "tax_rate", m.replace('％', '%'))

    # =========================
    # 後処理（正規化・統合）
    # =========================
    def _normalize_fields(self, fields: Dict[str, List[str]]):
        # 金額のノーマライズ
        def norm_money(s: str) -> str:
            s = s.replace(' ', '')
            s = s.replace('￥', '¥')
            s = re.sub(r'-$', '', s)
            return s

        fields['money'] = [norm_money(m) for m in fields['money']]
        if fields['money'] and not fields['total_amount']:
            fields['total_amount'].append(fields['money'][0])

        # 税率は % 統一
        fields['tax_rate'] = [r.replace('％', '%') for r in fields['tax_rate']]

        # org と company を統合（必ずリスト）
        orgs = fields.get("org", [])
        comps = fields.get("company", [])
        merged = []
        seen = set()
        for v in orgs + comps:
            v = v.strip()
            if v and v not in seen:
                merged.append(v)
                seen.add(v)
        fields["company"] = merged
        fields["org"] = []  # 以後未使用

        # registration_number: T13桁のみを残す
        cleaned_nums = []
        seen = set()
        for rn in fields["registration_number"]:
            m = re.search(r'(T\d{13})', rn)
            if m:
                core = m.group(1)
                if core not in seen:
                    cleaned_nums.append(core)
                    seen.add(core)
        fields["registration_number"] = cleaned_nums

    # =========================
    # フォーマット
    # =========================
    def _format_result(self, fields: Dict[str, List[str]], texts: List[str]) -> Dict[str, Any]:
        print("\n[ステップ3] 結果を整形中...")

        summary = {
            "date": fields["date"],
            "company": fields["company"],
            "tax_rate": fields["tax_rate"],
            "total_amount": fields["total_amount"],
            "registration_number": fields["registration_number"],
            "registration": bool(fields["registration_number"]),
        }

        return {
            "status": "success",
            "summary": summary,
            "raw_texts": texts,
        }


# =========================
# 実行例
# =========================
def main():
    try:
        parser = ReceiptParserUnified()
        result = parser.parse_receipt("2.jpg")
        print("\n" + "=" * 50)
        print("解析結果")
        print("=" * 50)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"致命的エラー: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()