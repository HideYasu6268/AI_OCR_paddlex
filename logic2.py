import spacy
from paddleocr import PaddleOCR
import json
import re
import traceback
import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class ReceiptParserTransformers:
    def __init__(self):
        """初期化: OCRエンジンとTransformersモデルのロード"""
        try:
            # PaddleOCRの初期化（最小限のパラメータ）
            self.ocr = PaddleOCR(lang='japan')
            
            # Transformersモデルのロード
            # ★★★ 唯一の変更点 ★★★
            model_path = r"C:\MyPython\AI_OCR_naver\models\cl-tohokubert-base-japanese-v2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            print("✓ モデルの初期化に成功しました (PaddleOCR + Transformers NER)")
        except Exception as e:
            print(f"✗ モデルの初期化に失敗しました: {e}")
            raise

    def parse_receipt(self, image_path: str) -> Dict[str, Any]:
        """
        領収書画像を解析して構造化データを返す
        
        Args:
            image_path (str): 領収書画像のパス
            
        Returns:
            Dict[str, Any]: 解析結果の辞書
        """
        print(f"\n{'='*50}")
        print(f"領収書解析を開始: {image_path}")
        print("="*50)
        
        # 画像ファイルの存在確認
        if not os.path.exists(image_path):
            error_msg = f"画像ファイルが見つかりません: {image_path}"
            print(f"エラー: {error_msg}")
            return {"error": error_msg, "raw_texts": []}
        
        try:
            # 1. OCR処理
            texts = self._run_ocr(image_path)
            if not texts:
                return {"error": "テキストが抽出できませんでした", "raw_texts": []}
            
            # 2. エンティティ抽出
            entities = self._extract_entities(texts)
            
            # 3. 結果整形
            return self._format_result(entities, texts)
            
        except Exception as e:
            print(f"解析中にエラーが発生しました: {e}")
            traceback.print_exc()
            return {"error": str(e), "raw_texts": []}

    def _run_ocr(self, image_path: str) -> List[str]:
        """OCR処理の実行"""
        print("\n[ステップ1] OCR処理を実行中...")
        
        try:
            # 最新APIの呼び出し（パラメータなし）
            result = self.ocr.predict(image_path)
            texts = self._extract_texts(result)
            
            print(f"抽出されたテキスト ({len(texts)}件):")
            for i, text in enumerate(texts, 1):
                print(f"  {i:2d}: {text}")
            
            return texts
            
        except Exception as e:
            print(f"OCR処理に失敗しました: {e}")
            raise

    def _extract_texts(self, ocr_result: Any) -> List[str]:
        """OCR結果からテキストを抽出"""
        texts = []
        
        try:
            # 最新のPaddleOCRの結果構造に対応
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                page_result = ocr_result[0]
                
                # OCRResultオブジェクトの場合 - 直接属性アクセスを試行
                try:
                    rec_texts = page_result.rec_texts
                    print(f"rec_textsが見つかりました: {type(rec_texts)}, 長さ: {len(rec_texts)}")
                    
                    if isinstance(rec_texts, list):
                        for text in rec_texts:
                            if text and str(text).strip():  # 空文字列を除外
                                clean_text = str(text).strip()
                                texts.append(clean_text)
                                print(f"  テキスト抽出: {clean_text}")
                    else:
                        print(f"rec_textsが予期しない型: {type(rec_texts)}")
                        
                except AttributeError:
                    print("rec_texts属性が見つかりません")
                    
                    # 辞書アクセスを試行
                    try:
                        if hasattr(page_result, '__getitem__'):
                            rec_texts = page_result['rec_texts']
                            print(f"辞書アクセスでrec_textsが見つかりました: {len(rec_texts)}件")
                            
                            for text in rec_texts:
                                if text and str(text).strip():
                                    clean_text = str(text).strip()
                                    texts.append(clean_text)
                                    print(f"  テキスト抽出: {clean_text}")
                    except (KeyError, TypeError):
                        print("辞書アクセスも失敗")
                        
                        # 従来のリスト形式を試行
                        if isinstance(page_result, list):
                            print("従来のリスト形式を試行")
                            for line in page_result:
                                if line and len(line) >= 2:
                                    text_info = line[1]
                                    if isinstance(text_info, (tuple, list)) and len(text_info) >= 1:
                                        text = str(text_info[0]).strip()
                                        if text:
                                            texts.append(text)
                                            print(f"  テキスト抽出: {text}")
                
                except Exception as e:
                    print(f"属性アクセス中の予期しないエラー: {e}")
                    traceback.print_exc()
                    
            else:
                print("OCR結果が空または予期しない形式です")
                
        except Exception as e:
            print(f"テキスト抽出中にエラー: {e}")
            traceback.print_exc()
        
        print(f"最終的に抽出されたテキスト数: {len(texts)}")
        return texts

    def _extract_entities(self, texts: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """エンティティ抽出"""
        print("\n[ステップ2] エンティティ抽出を実行中...")
        
        # TransformersによるNER
        transformers_entities = self._transformers_ner(texts)
        
        # カスタムルールによる抽出
        custom_entities = self._custom_ner(texts)
        
        return {
            "transformers": transformers_entities,
            "custom": custom_entities
        }

    def _transformers_ner(self, texts: List[str]) -> Dict[str, List[str]]:
        """Transformers NERモデルでエンティティを抽出（改善版）"""
        print("- Transformers NERを実行...")
        entities = {
            'PERSON': [], 'ORG': [], 'DATE': [],
            'MONEY': [], 'QUANTITY': []
        }
        
        # 1. テキスト前処理
        processed_texts = []
        for text in texts:
            # 不要な記号や空白を除去
            cleaned = re.sub(r'[　\s]+', ' ', text.strip())  # 全角/半角スペースを正規化
            cleaned = re.sub(r'[、。・]', ' ', cleaned)  # 句読点をスペースに
            if cleaned:
                processed_texts.append(cleaned)
        
        # 2. 適切な長さのチャンクに分割（モデルの最大長を考慮）
        max_length = 512  # モデルの最大トークン長
        chunks = []
        current_chunk = ""
        
        for text in processed_texts:
            if len(current_chunk) + len(text) + 1 < max_length:
                current_chunk += " " + text if current_chunk else text
            else:
                chunks.append(current_chunk)
                current_chunk = text
        if current_chunk:
            chunks.append(current_chunk)
        
        # 3. NER実行（バッチ処理とパラメータ調整）
        ner_results = []
        for chunk in chunks:
            try:
                results = self.ner_pipeline(
                    chunk,
                    batch_size=4,  # バッチサイズを調整
                    stride=32,     # オーバーラップ処理
                    ignore_labels=[],  # 全てのエンティティを考慮
                    device=0 if torch.cuda.is_available() else -1  # GPU利用
                )
                ner_results.extend(results)
            except Exception as e:
                print(f"NER処理中にエラー: {e}")
                continue
        
        # 4. エンティティの後処理
        seen_entities = set()  # 重複排除用
        
        for entity in ner_results:
            entity_type = entity['entity_group']
            entity_text = entity['word'].strip()
            
            # エンティティタイプの正規化
            if entity_type not in entities:
                if entity_type in ['LOC', 'GPE']:
                    entity_type = 'ORG'
                else:
                    continue
            
            # 無意味な単語をフィルタリング
            if len(entity_text) < 2:
                continue
            if entity_text.isdigit():
                continue
                
            # 重複排除
            entity_key = f"{entity_type}_{entity_text}"
            if entity_key not in seen_entities:
                entities[entity_type].append(entity_text)
                seen_entities.add(entity_key)
                print(f"  [Transformers] {entity_text} → {entity_type}")
        
        # 5. 金額と日付の特別処理
        for text in processed_texts:
            # 金額の補足検出
            money_matches = re.findall(r'[￥¥]\s*\d{1,3}(?:,\d{3})*', text)
            for match in money_matches:
                if match not in entities['MONEY']:
                    entities['MONEY'].append(match)
                    print(f"  [補足] {match} → MONEY")
            
            # 日付の補足検出
            date_matches = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}', text)
            for match in date_matches:
                if match not in entities['DATE']:
                    entities['DATE'].append(match)
                    print(f"  [補足] {match} → DATE")
        
        return entities
    def _custom_ner(self, texts: List[str]) -> Dict[str, List[str]]:
        """カスタムルールで領収書情報を抽出"""
        print("- カスタムルールを適用...")
        patterns = {
            'RECEIPT_DATE': [
                r'\d{4}/\d{1,2}/\d{1,2}',
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{2}/\d{1,2}/\d{1,2}',
                r'令和\d+年\d+月\d+日'
            ],
            'RECEIPT_AMOUNT': [
                r'￥[\d,]+',
                r'¥[\d,]+',
                r'税込\s*￥?[\d,]+',
                r'税抜\s*￥?[\d,]+',
                r'小計\s*￥?[\d,]+',
                r'合計\s*￥?[\d,]+',
                r'金額\s*￥?[\d,]+'
            ],
            'RECEIPT_TAX': [
                r'消費税\s*￥?[\d,]+',
                r'内税\s*￥?[\d,]+',
                r'税額\s*￥?[\d,]+',
                r'税\s*￥?[\d,]+'
            ],
            'RECEIPT_COMPANY': [
                r'.+店',
                r'.+株式会社',
                r'.+有限会社',
                r'.+Co\.',
                r'.+Ltd\.',
                r'.+商店'
            ],
            'RECEIPT_NUMBER': [
                r'No\.?\s*\d+',
                r'領収№\s*\d+',
                r'レシート№\s*\d+',
                r'受付番号\s*\d+',
                r'管理番号\s*\d+'
            ],
            'RECEIPT_REGISTRATION': [
                r'T\d{13}',
                r'登録番号\s*T?\d{13}'
            ],
            'RECEIPT_TAX_RATE': [
                r'\d+%',
                r'\d+％'
            ]
        }
        
        entities = {k: [] for k in patterns.keys()}
        
        for text in texts:
            text = text.strip()
            if not text:
                continue
                
            for entity_type, regex_list in patterns.items():
                for pattern in regex_list:
                    matches = re.findall(pattern, text)
                    if matches:
                        for match in matches:
                            if match not in entities[entity_type]:
                                entities[entity_type].append(match)
                                print(f"  [Custom] {match} → {entity_type}")
        
        return entities

    def _format_result(
        self,
        entities: Dict[str, Dict[str, List[str]]],
        texts: List[str]
    ) -> Dict[str, Any]:
        """結果を整形"""
        print("\n[ステップ3] 結果を整形中...")
        
        summary = {
            'date': self._get_first_value(entities['custom']['RECEIPT_DATE']),
            'total_amount': self._get_first_value(entities['custom']['RECEIPT_AMOUNT']),
            'tax_amount': self._get_first_value(entities['custom']['RECEIPT_TAX']),
            'company': self._get_first_value(entities['custom']['RECEIPT_COMPANY']),
            'receipt_no': self._get_first_value(entities['custom']['RECEIPT_NUMBER']),
            'registration': self._get_first_value(entities['custom']['RECEIPT_REGISTRATION']),
            'tax_rate': self._get_first_value(entities['custom']['RECEIPT_TAX_RATE'])
        }
        
        return {
            'status': 'success',
            'extraction_method': 'Transformers NER + Custom Rules',
            'entities': entities,
            'summary': summary,
            'raw_texts': texts
        }

    def _get_first_value(self, values: List[str]) -> Optional[str]:
        """リストから最初の値を取得"""
        return values[0] if values else None


def main():
    """メイン実行関数"""
    try:
        parser = ReceiptParserTransformers()
        result = parser.parse_receipt("2.jpg")
        
        print("\n" + "="*50)
        print("解析結果")
        print("="*50)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"致命的なエラーが発生しました: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()