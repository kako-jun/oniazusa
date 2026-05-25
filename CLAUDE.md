# oniazusa

写真を `痕` 風ノベル背景へ寄せる小さな Python CLI。`skirts-colour` の背景加工実験を、
大きな本体リポジトリから切り出して回すための検証場であり、将来的には `name-name`
や他の画像ツールから呼ばれる基盤でもあります。

## ドキュメント

| ファイル | 内容 | 言語 |
|---|---|---|
| `README.md` | エンドユーザー向け概要と使い方 | 英語 |
| `docs/overview.md` | 目指す画作りと非目標 | 英語 |
| `docs/spec.md` | CLI とフィルタ処理仕様 | 英語 |
| `docs/roadmap.md` | 品質課題と改善順序 | 日本語 |
| `CLAUDE.md` | AI 向け内部メモ | 日本語 |

## 現在の構造

```
src/oniazusa/
├── cli.py      # argparse 入口
└── filter.py   # 痕風フィルタ本体

tests/
├── test_cli.py
└── test_filter.py
```

## 開発コマンド

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
uv run oniazusa --help
```

## 設計メモ

- 目的は「汎用アニメ化」ではなく、痕風の背景らしさに寄せること
- 他ツールから使う前提なので、CLI 契約と Python API の安定性も成果物に含む
- 色味、階調、ドット感、前景/背景の扱いを分けて考える
- PC-98 的な格子状グラデーションは ordered dithering / Bayer dithering として扱う
- 順序は「なだらかな階調づくり → 格子ディザ」が基本で、逆では崩れやすい
- 現状は全画面を一律処理しているため、狙い通りにならない写真が多い
- MiDaS などの深度推定は、品質改善の本命候補

## 実装ルール

- 新しい見た目パラメータを入れたら synthetic test か比較用サンプルを用意する
- 参考画面との比較メモを `docs/roadmap.md` か Issue に残す
- Python 実行は `uv run python3` を使う
