# oniazusa Roadmap

## 現状

- 最小の痕風フィルタは動く
- README / docs / CI / tests は整備済み
- しかし「痕らしさ」の詰めはまだ粗い
- `docs/research-notes.md` に PC-98 側の制約と ordered dithering の前提メモを残した
- `input/ear-sky/NOTES.md` に夜景系 input の見るべき点を残した
- 方針: まずは **PC-98 忠実寄り** を本命にする。派生的に現代寄りの綺麗さを選べてもよい

## 近い課題

- MiDaS 深度推定で前景/背景の処理を分ける
- ティント色を実画面比較で追い込む
- 屋外だけでなく室内、街並み、逆光でも破綻しないか検証する
- PC-98 的な格子グラデーションを reference 比較で詰める

## 推奨着手順

1. **Issue #7: explicit 3-tone background mode**
   - まず土台のトーン設計を固定する
2. **Issue #5: outline-strategy comparison workflow**
   - 最濃色を edge にどう配るかを比較実験で決める
3. **Issue #6: tint palette calibration**
   - `input/kizuato` を見ながら明部/暗部/最濃部の色関係を詰める
4. **Issue #4: preprocess modes**
   - scene ごとの差を吸収する前処理の分岐を足す
5. **Issue #1: MiDaS depth-aware processing**
   - ここまでで 2D 側の土台を固めてから depth に入る
6. **Issue #3: reusable quality harness**
   - 比較実験で見えてきた評価軸を固定して保守基盤にする

## 優先順位の考え方

- 先に **tone structure** を決める
- 次に **outline / tint** を詰める
- その後で **scene-specific preprocess** と **depth-aware split** に進む
- quality harness は最後ではなく並行で育てるが、評価軸が見えてから固める

## モード方針

- default は PC-98 忠実寄り
- ただし将来的に「少し現代的で綺麗」な派生モードを追加してもよい
- まずは忠実寄りの基準を作らないと派生モードも定義できない

## 予定している拡張

- `--preprocess` で写真からイラスト寄りに振る複数経路
- 参考画像セットを使った見た目回帰テスト
- パラメータ探索のための比較コラージュ出力
- 他ツールから安全に呼べる API / exit code / 出力規約の固定
