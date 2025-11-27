# -*- coding: utf-8 -*-
import os
import re
import csv
import traceback
from typing import List, Dict, Any, Optional

# 依存
import torch
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


# ==============================
# パーサ（必要項目のみ）
# ==============================
class ReceiptParserUnified:
    """
    返すのは:
      - date (list[str])
      - company (list[str])  ← NER(org)＋カスタム(company)を統合
      - tax_rate (list[str])
      - total_amount (list[str])
      - registration_number (list[str])  ← 「T+13桁」限定
      - registration (bool) ← registration_number が1つ以上あれば True
    """

    def __init__(self):
        try:
            self.ocr = PaddleOCR(lang='japan')
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

    def parse_receipt(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"error": f"画像ファイルが見つかりません: {image_path}", "raw_texts": []}
        try:
            texts = self._run_ocr(image_path)
            if not texts:
                return {"error": "テキストが抽出できませんでした", "raw_texts": []}
            fields = self._empty_fields()
            self._apply_transformers_ner(texts, fields)
            self._apply_custom_rules(texts, fields)
            self._normalize_fields(fields)
            return self._format_result(fields, texts)
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e), "raw_texts": []}

    # --- OCR ---
    def _run_ocr(self, image_path: str) -> List[str]:
        result = self.ocr.predict(image_path)
        return self._extract_texts(result)

    def _extract_texts(self, ocr_result: Any) -> List[str]:
        texts: List[str] = []
        try:
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                page = ocr_result[0]
                try:
                    rec_texts = page.rec_texts
                    for s in rec_texts:
                        s = str(s).strip()
                        if s: texts.append(s)
                    return texts
                except AttributeError:
                    pass
                try:
                    if hasattr(page, '__getitem__'):
                        rec_texts = page['rec_texts']
                        for s in rec_texts:
                            s = str(s).strip()
                            if s: texts.append(s)
                        return texts
                except Exception:
                    pass
                if isinstance(page, list):
                    for line in page:
                        if line and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, (tuple, list)) and len(text_info) >= 1:
                                s = str(text_info[0]).strip()
                                if s: texts.append(s)
        except Exception as e:
            print(f"OCRテキスト抽出エラー: {e}")
            traceback.print_exc()
        return texts

    # --- 統一フィールド ---
    def _empty_fields(self) -> Dict[str, List[str]]:
        return {
            "date": [],
            "company": [],
            "tax_rate": [],
            "total_amount": [],
            "registration_number": [],
            # 内部補助
            "money": [],
            "org": [],
        }

    def _add(self, fields: Dict[str, List[str]], key: str, value: str):
        v = value.strip()
        if v and v not in fields[key]:
            fields[key].append(v)

    # --- NER ---
    def _apply_transformers_ner(self, texts: List[str], fields: Dict[str, List[str]]):
        try:
            cleaned = []
            for t in texts:
                t = re.sub(r'[　\s]+', ' ', t.strip())
                t = re.sub(r'[、。・]', ' ', t)
                if t: cleaned.append(t)
            joined = " ".join(cleaned)[:3000]
            results = self.ner(joined)
            for ent in results:
                label = ent.get("entity_group", "")
                word = ent.get("word", "").strip()
                if not word or len(word) < 2: continue
                if label in ("DATE",):
                    self._add(fields, "date", word)
                elif label in ("MONEY",):
                    self._add(fields, "money", word)
                elif label in ("ORG", "GPE", "LOC"):
                    self._add(fields, "org", word)

            for t in cleaned:
                for m in re.findall(r'[￥¥]\s*\d{1,3}(?:,\d{3})*(?:-\s*)?', t):
                    self._add(fields, "money", m.replace(" ", ""))
                for d in re.findall(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}', t):
                    self._add(fields, "date", d)
                for rate in re.findall(r'(?<!\d)(?:\d{1,2})[%％](?!\d)', t):
                    self._add(fields, "tax_rate", rate.replace('％', '%'))
                for regnum in re.findall(r'T\d{13}', t):
                    self._add(fields, "registration_number", regnum)
        except Exception:
            traceback.print_exc()

    # --- カスタムルール ---
    def _apply_custom_rules(self, texts: List[str], fields: Dict[str, List[str]]):
        for s in texts:
            s = s.strip()
            if not s: continue
            for pat in [r'\d{4}/\d{1,2}/\d{1,2}', r'\d{4}年\d{1,2}月\d{1,2}日',
                        r'\d{2}/\d{1,2}/\d{1,2}', r'令和\d+年\d+月\d+日']:
                for m in re.findall(pat, s):
                    self._add(fields, "date", m)
            for pat in [r'￥\s*[\d,]+-?', r'¥\s*[\d,]+-?']:
                for m in re.findall(pat, s):
                    self._add(fields, "money", m.replace(" ", ""))
            for pat in [r'.+店', r'.+株式会社', r'.+有限会社', r'.+Co\.', r'.+Ltd\.', r'.+商店']:
                m = re.findall(pat, s)
                for v in m:
                    self._add(fields, "company", v.strip('「」[]<>'))
            for regnum in re.findall(r'T\d{13}', s):
                self._add(fields, "registration_number", regnum)
            for m in re.findall(r'\d{1,2}[%％]', s):
                self._add(fields, "tax_rate", m.replace('％', '%'))

    # --- 後処理 ---
    def _normalize_fields(self, fields: Dict[str, List[str]]):
        def norm_money(s: str) -> str:
            s = s.replace(' ', '').replace('￥', '¥')
            s = re.sub(r'-$', '', s)
            return s
        fields['money'] = [norm_money(m) for m in fields['money']]
        if fields['money'] and not fields['total_amount']:
            fields['total_amount'].append(fields['money'][0])
        fields['tax_rate'] = [r.replace('％', '%') for r in fields['tax_rate']]

        orgs = fields.get("org", [])
        comps = fields.get("company", [])
        merged, seen = [], set()
        for v in orgs + comps:
            v = v.strip()
            if v and v not in seen:
                merged.append(v); seen.add(v)
        fields["company"] = merged
        fields["org"] = []

        cleaned_nums, seen = [], set()
        for rn in fields["registration_number"]:
            m = re.search(r'(T\d{13})', rn)
            if m:
                core = m.group(1)
                if core not in seen:
                    cleaned_nums.append(core); seen.add(core)
        fields["registration_number"] = cleaned_nums

    def _format_result(self, fields: Dict[str, List[str]], texts: List[str]) -> Dict[str, Any]:
        summary = {
            "date": fields["date"],
            "company": fields["company"],
            "tax_rate": fields["tax_rate"],
            "total_amount": fields["total_amount"],
            "registration_number": fields["registration_number"],
            "registration": bool(fields["registration_number"]),
        }
        return {"status": "success", "summary": summary, "raw_texts": texts}


# ==================================
# Tkinter デスクトップアプリ
# （全件一括AIOCR→各画像で確認→最後にCSV保存 / 表記：インボイス登録 / 手入力は画像ごとに独立）
# ==================================
class ReceiptApp(tk.Tk):
    CSV_COLUMNS = ["date", "company", "tax_rate", "total_amount", "registration_number", "インボイス登録"]

    def __init__(self):
        super().__init__()
        self.title("Receipt Parser Desktop")
        self.geometry("1000x700")
        self.minsize(900, 620)

        self.parser = ReceiptParserUnified()
        self.image_paths: List[str] = []
        self.current_index: int = -1

        self.cache_results: Dict[str, Dict[str, Any]] = {}
        self.saved_rows: Dict[str, Dict[str, str]] = {}
        self.manual_inputs: Dict[str, Dict[str, set]] = {}

        self.csv_path: str = ""
        self.dirty: bool = False

        self._build_ui()
        self._update_nav_buttons()

    # ---- ポップアップ ----
    def _processing_popup(self, text: str, determinate: bool = False, maximum: int = 100):
        win = tk.Toplevel(self)
        win.title("AIOCR")  # タイトル
        win.transient(self)
        win.attributes("-topmost", True)
        win.grab_set()
        frm = ttk.Frame(win, padding=16)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=text, font=("", 12)).pack(pady=(0, 8))
        mode = "determinate" if determinate else "indeterminate"
        pb = ttk.Progressbar(frm, mode=mode, maximum=maximum)
        pb.pack(fill="x")
        if determinate:
            pb["value"] = 0
        else:
            pb.start(10)
        # 中央配置
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 200
        y = self.winfo_y() + (self.winfo_height() // 2) - 50
        win.geometry(f"400x100+{x}+{y}")
        self.update_idletasks()
        return win, pb

    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="画像を開く（複数可）", command=self.on_open).pack(side="left")
        # ← ナビボタンは参照を保持して state を切替
        self.btn_prev = ttk.Button(top, text="前へ", command=self.on_prev)
        self.btn_prev.pack(side="left", padx=6)
        self.btn_next = ttk.Button(top, text="次へ", command=self.on_next)
        self.btn_next.pack(side="left")

        right_btns = ttk.Frame(top); right_btns.pack(side="right")
        ttk.Button(right_btns, text="全て保存", command=self.on_save_all).pack(side="right")

        main = ttk.Frame(self); main.pack(fill="both", expand=True, padx=8, pady=6)

        # 左：画像＋raw_texts（タイトル付き）
        left = ttk.Frame(main); left.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="画像プレビュー", font=("", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.canvas = tk.Canvas(left, bg="#f2f2f2", height=360)
        self.canvas.pack(fill="x")

        ttk.Label(left, text="読取結果", font=("", 10, "bold")).pack(anchor="w", pady=(8, 2))
        self.raw_list = tk.Listbox(left, height=12)
        self.raw_list.pack(fill="both", expand=True, pady=(0, 0))

        # 右：ドロップダウン（手入力可・画像ごと独立）
        right = ttk.Frame(main); right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.var_date = tk.StringVar()
        self.var_company = tk.StringVar()
        self.var_tax_rate = tk.StringVar()
        self.var_total = tk.StringVar()
        self.var_regnum = tk.StringVar()
        self.var_registration = tk.BooleanVar(value=False)

        for v in [self.var_date, self.var_company, self.var_tax_rate, self.var_total, self.var_regnum]:
            v.trace_add("write", lambda *_: self._mark_dirty())
        self.var_registration.trace_add("write", lambda *_: self._mark_dirty())

        grid = ttk.Frame(right); grid.pack(fill="x", pady=8)

        def add_row(row, label, widget):
            ttk.Label(grid, text=label, width=20, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            grid.columnconfigure(1, weight=1)

        self.cb_date = ttk.Combobox(grid, textvariable=self.var_date, values=[])
        add_row(0, "日付 (date)", self.cb_date)

        self.cb_company = ttk.Combobox(grid, textvariable=self.var_company, values=[])
        add_row(1, "会社/店舗 (company)", self.cb_company)

        self.cb_tax_rate = ttk.Combobox(grid, textvariable=self.var_tax_rate, values=[])
        add_row(2, "税率 (tax_rate)", self.cb_tax_rate)

        self.cb_total = ttk.Combobox(grid, textvariable=self.var_total, values=[])
        add_row(3, "合計 (total_amount)", self.cb_total)

        self.cb_regnum = ttk.Combobox(grid, textvariable=self.var_regnum, values=[])
        add_row(4, "登録番号 (T13桁)", self.cb_regnum)

        reg_row = ttk.Frame(right); reg_row.pack(fill="x", pady=(8, 0))
        ttk.Label(reg_row, text="インボイス登録（T13桁が入ればON）").pack(side="left")
        ttk.Checkbutton(reg_row, variable=self.var_registration).pack(side="left", padx=8)

        btns = ttk.Frame(right); btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="この画像を再解析", command=self.reparse_current).pack(side="left")

        bottom = ttk.Frame(self); bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.status = tk.StringVar(value="準備完了")
        ttk.Label(bottom, textvariable=self.status, anchor="w").pack(fill="x")

        # 手入力可にする（画像ごとに独立して保持）
        self._make_editable("date", self.cb_date,   self.var_date)
        self._make_editable("company", self.cb_company, self.var_company)
        self._make_editable("tax_rate", self.cb_tax_rate, self.var_tax_rate)
        self._make_editable("total_amount", self.cb_total,  self.var_total)
        self._make_editable("registration_number", self.cb_regnum, self.var_regnum, filter_t13=True)

        # 登録番号変更 → インボイス登録 自動同期（IME確定含む）
        def _sync_registration_on_regnum(*_):
            v = self.var_regnum.get().strip()
            self.var_registration.set(bool(re.fullmatch(r"T\d{13}", v)))
        self.var_regnum.trace_add("write", _sync_registration_on_regnum)

    # ---- 編集可能コンボ（画像ごと独立で保持）----
    def _make_editable(self, key: str, cb: ttk.Combobox, var: tk.StringVar, *, filter_t13: bool = False):
        cb.configure(state="normal")
        def commit(event=None):
            s = var.get().strip()
            if not s:
                return
            if filter_t13 and not re.fullmatch(r"T\d{13}", s):
                messagebox.showwarning("登録番号の形式", "登録番号は「T」+13桁のみ有効です。例: T1234567890123")
                return "break"
            # 現在の画像にのみ手動候補を記録
            if self.image_paths and self.current_index >= 0:
                path = self.image_paths[self.current_index]
                self.manual_inputs.setdefault(path, {}).setdefault(key, set()).add(s)
            self._mark_dirty()
            if filter_t13:
                self.var_registration.set(bool(re.fullmatch(r"T\d{13}", s)))
            # 現在画像の候補を再構築（他画像には影響しない）
            self.populate_widgets()
            # 入力値を選択状態に保持
            var.set(s)
        cb.bind("<Return>", commit)
        cb.bind("<FocusOut>", commit)

    # ---- ナビボタン状態更新 ----
    def _update_nav_buttons(self):
        if not self.image_paths:
            self.btn_prev.state(["disabled"])
            self.btn_next.state(["disabled"])
            return
        n = len(self.image_paths)
        i = max(0, min(self.current_index, n - 1))
        # 最初の画像 → 前へ 無効
        if i <= 0:
            self.btn_prev.state(["disabled"])
        else:
            self.btn_prev.state(["!disabled"])
        # 最後の画像 → 次へ 無効
        if i >= n - 1:
            self.btn_next.state(["disabled"])
        else:
            self.btn_next.state(["!disabled"])

    # ---- 状態管理 ----
    def _mark_dirty(self):
        self.dirty = True
        self.status.set("未保存の変更があります（内容を確定してください）")

    def _clear_dirty(self):
        self.dirty = False

    # ---- ファイル選択 → 一括AIOCR ----
    def on_open(self):
        if not self._maybe_confirm_and_buffer():
            return
        paths = filedialog.askopenfilenames(
            title="領収書画像を選択",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not paths:
            return

        self.image_paths = list(paths)
        self.current_index = 0
        self.saved_rows.clear()
        self.cache_results.clear()
        self.manual_inputs.clear()  # 手動候補もリセット
        self._update_nav_buttons()

        self._preprocess_all(self.image_paths)   # 全件一括AIOCR
        self.load_current()                      # 以降はキャッシュを表示
        self._update_nav_buttons()

    def _preprocess_all(self, paths: List[str]):
        dlg, pb = self._processing_popup("全画像をAIOCR 処理中…", determinate=True, maximum=max(1, len(paths)))
        try:
            for i, p in enumerate(paths, 1):
                self.status.set(f"AIOCR 処理中 {i}/{len(paths)}: {os.path.basename(p)}")
                self.update_idletasks()
                try:
                    res = self.parser.parse_receipt(p)
                except Exception as e:
                    res = {"status": "error", "error": str(e), "summary": {}, "raw_texts": []}
                self.cache_results[p] = res
                pb["value"] = i
                self.update_idletasks()
        finally:
            try:
                dlg.destroy()
            except Exception:
                pass
        self.status.set("AIOCR 処理完了。各画像を確認してください。")

    # ---- 画面移動（循環しない / 端で無効）----
    def on_prev(self):
        if not self.image_paths or self.current_index <= 0:
            return
        if not self._maybe_confirm_and_buffer():
            return
        self.current_index -= 1
        self.load_current()
        self._update_nav_buttons()

    def on_next(self):
        if not self.image_paths or self.current_index >= len(self.image_paths) - 1:
            return
        if not self._maybe_confirm_and_buffer():
            return
        self.current_index += 1
        self.load_current()
        self._update_nav_buttons()

    def _maybe_confirm_and_buffer(self) -> bool:
        """未確定(=dirty or 未保存)なら確認してバッファに確定。キャンセル時は移動中止。"""
        if not self.image_paths or self.current_index < 0:
            return True
        path = self.image_paths[self.current_index]
        need = self.dirty or (path not in self.saved_rows)
        if not need:
            return True
        row = self._confirm_and_build_row()
        if row is None:
            return messagebox.askyesno("未保存の変更", "未保存の変更があります。破棄して移動しますか？")
        self.saved_rows[path] = row  # 確定
        self._clear_dirty()
        return True

    def load_current(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]

        # 画像プレビューのみ（OCRはしない）
        self.show_image(path)

        # キャッシュから結果を取得
        self.current_result = self.cache_results.get(path, {"summary": {}, "raw_texts": []})
        self.populate_widgets()

        # 既にバッファに確定行があればUIに反映
        if path in self.saved_rows:
            row = self.saved_rows[path]
            self.var_date.set(row["date"])
            self.var_company.set(row["company"])
            self.var_tax_rate.set(row["tax_rate"])
            self.var_total.set(row["total_amount"])
            self.var_regnum.set(row["registration_number"])
            self.var_registration.set(row["registration"] == "TRUE")
        else:
            # 登録番号に同期
            self.var_registration.set(bool(re.fullmatch(r"T\d{13}", self.var_regnum.get().strip())))

        self._clear_dirty()
        self.status.set(f"{os.path.basename(path)}（AIOCR済み）を表示中。確認後は前/次で移動、最後に『全て保存』。")
        self._update_nav_buttons()

    # ---- 個別再解析（キャッシュ更新）----
    def reparse_current(self):
        if not self.image_paths:
            return
        if self.dirty and not messagebox.askyesno("未保存の変更", "変更を破棄して再解析しますか？"):
            return
        path = self.image_paths[self.current_index]
        dlg, _ = self._processing_popup("AIOCR 処理中…")
        try:
            self.update_idletasks()
            res = self.parser.parse_receipt(path)
            self.cache_results[path] = res
        finally:
            try:
                dlg.destroy()
            except Exception:
                pass
        self.load_current()
        self._update_nav_buttons()

    # ---- 候補生成 & UI反映（画像ごとの手動候補を混ぜる）----
    def _candidates(self, summary: Dict[str, Any], raw_texts: List[str], key: str) -> List[str]:
        seen = set(); vals = []
        for v in summary.get(key, []):
            s = str(v).strip()
            if s and s not in seen:
                vals.append(s); seen.add(s)
        for t in raw_texts:
            s = str(t).strip()
            if s and s not in seen:
                vals.append(s); seen.add(s)
        # 手動候補（この画像だけ）
        if self.image_paths and self.current_index >= 0:
            path = self.image_paths[self.current_index]
            for s in self.manual_inputs.get(path, {}).get(key, set()):
                s = str(s).strip()
                if s and s not in seen:
                    vals.append(s); seen.add(s)
        return vals or [""]

    def populate_widgets(self):
        res = self.current_result or {}
        summary = res.get("summary", {})
        raw_texts = res.get("raw_texts", [])

        self.raw_list.delete(0, tk.END)
        for t in raw_texts:
            self.raw_list.insert(tk.END, t)

        def set_combo(cb: ttk.Combobox, var: tk.StringVar, key: str, filter_t13=False):
            items = self._candidates(summary, raw_texts, key)
            if filter_t13:
                items = [c for c in items if re.fullmatch(r"T\d{13}", c)] or [""]
            cb["values"] = items
            cur = var.get().strip()
            if cur in items:
                cb.set(cur)
            else:
                var.set(items[0] if items and items[0] != "" else "")

        set_combo(self.cb_date, self.var_date, "date")
        set_combo(self.cb_company, self.var_company, "company")
        set_combo(self.cb_tax_rate, self.var_tax_rate, "tax_rate")
        set_combo(self.cb_total, self.var_total, "total_amount")
        set_combo(self.cb_regnum, self.var_regnum, "registration_number", filter_t13=True)

    # ---- 確認 & 行構築 ----
    def _confirm_and_build_row(self) -> Optional[Dict[str, str]]:
        row = {
            "date": self.var_date.get().strip(),
            "company": self.var_company.get().strip(),
            "tax_rate": self.var_tax_rate.get().strip(),
            "total_amount": self.var_total.get().strip(),
            "registration_number": self.var_regnum.get().strip(),
            "registration": "TRUE" if self.var_registration.get() else "FALSE",
        }
        msg = (
            "以下の内容で 確認済み とします。よろしいですか？\n\n"
            f"日付: {row['date']}\n"
            f"会社/店舗: {row['company']}\n"
            f"税率: {row['tax_rate']}\n"
            f"合計: {row['total_amount']}\n"
            f"登録番号(T13桁): {row['registration_number']}\n"
            f"インボイス登録: {row['registration']}"
        )
        if not messagebox.askokcancel("確認", msg):
            return None
        return row

    # ---- 金額のCSV出力用クレンジング（全角→半角＆数字以外除去）----
    def _normalize_amount_for_csv(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        # 全角数字→半角
        fw = "０１２３４５６７８９"
        hw = "0123456789"
        s = s.translate(str.maketrans(fw, hw))
        # 数字以外をすべて削除（¥, ￥, カンマ, スペース, 記号など）
        s = re.sub(r'[^0-9]', '', s)
        return s

    # ---- CSV行（ヘッダ名に合わせて変換）----
    def _row_to_csv_row(self, row: Dict[str, str]) -> Dict[str, str]:
        return {
            "date": row["date"],
            "company": row["company"],
            "tax_rate": row["tax_rate"],
            "total_amount": self._normalize_amount_for_csv(row["total_amount"]),
            "registration_number": row["registration_number"],
            "インボイス登録": row["registration"],
        }

    # ---- 全て保存（Shift-JIS / cp932）----
    def on_save_all(self):
        if not self.image_paths:
            messagebox.showinfo("保存", "先に画像を選択してください。")
            return

        # 現在画像が未確定なら確認して確定
        if not self._maybe_confirm_and_buffer():
            return

        # 選択順で集計
        rows_internal = [self.saved_rows[p] for p in self.image_paths if p in self.saved_rows]
        if len(rows_internal) < len(self.image_paths):
            messagebox.showwarning("未確定の画像があります",
                                   "未確定の画像があります。すべての画像で内容を確定してから保存してください。")
            return

        # 保存先の選択（未設定なら選ばせる）
        if not self.csv_path:
            default = os.path.join(os.path.expanduser("~"), "Desktop", "receipts.csv")
            path = filedialog.asksaveasfilename(
                title="CSV保存先（Shift-JIS）",
                defaultextension=".csv",
                initialfile=os.path.basename(default),
                filetypes=[("CSV", "*.csv")]
            )
            if not path:
                return
            self.csv_path = path

        # Shift-JIS（cp932）で上書き保存（ヘッダ=日本語「インボイス登録」）
        try:
            with open(self.csv_path, "w", newline="", encoding="cp932", errors="replace") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()
                writer.writerows([self._row_to_csv_row(r) for r in rows_internal])
            self._clear_dirty()
            messagebox.showinfo("保存完了", f"CSVを保存しました。\n{self.csv_path}\n（{len(rows_internal)} 行）")
            self.status.set("CSVを保存しました。必要なら修正後、再度『全て保存』で上書きできます。")
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))

    # ---- 画像プレビュー ----
    def show_image(self, path: str):
        try:
            img = Image.open(path)
            canvas_w = self.canvas.winfo_width() or 900
            max_h = 360
            w, h = img.size
            scale = min(canvas_w / max(1, w), max_h / max(1, h))
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_w // 2, max_h // 2, image=self.tk_img)
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor="nw", text=f"画像表示エラー: {e}")


# ---- main ----
if __name__ == "__main__":
    app = ReceiptApp()
    app.mainloop()