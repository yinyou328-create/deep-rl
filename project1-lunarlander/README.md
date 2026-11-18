# Project 1：用 DQN 玩 LunarLander-v3

以課本第 3 章的 DQN（Listing 3.5 experience replay、Listing 3.7 target network）為基礎，改寫成 Gymnasium 的 `LunarLander-v3`。

## 檔案

| 檔案 | 說明 |
|---|---|
| `Project1_LunarLander_DQN3.ipynb` | 最終版程式：每 25 局用 ε = 0 評估 50 局、保留最佳模型，訓練 1000 局 |
| `results/training_curves_improved_seed0.png`、`_seed1.png` | 訓練曲線（loss 與每局總分） |
| `results/test_scores_improved_seed0.png`、`_seed1.png` | 測試 100 局（ε = 0）每局分數 |
| `results/lunarlander_improved_seed0.gif` | seed 0 模型降落畫面 |

## 主要設定

| 項目 | 設定 |
|---|---|
| 網路 | 8 → 128 → 128 → 4，ReLU |
| gamma | 0.99 |
| epsilon | 1.0 → 0.01，每局 × 0.995 |
| Replay / batch | 100,000 / 64，累積 1,000 筆後開始訓練 |
| Target network | 每 1,000 步同步 |
| Loss / 梯度裁剪 | Huber（SmoothL1Loss）/ max norm 10 |
| 終止判斷 | 只用 `terminated`（被 1000 步截斷不算結束） |
| 評估機制 | 每 25 局用 ε = 0 評估 50 局，保留最佳權重，訓練結束後載回 |

## 結果（測試 100 局，ε = 0）

| seed | 平均分數 | 成功率（≥ 200 分） |
|---|---|---|
| 0 | 277.4 | 100% |
| 1 | 265.6 | 99% |
| 平均 | 271.5 | 99.5% |

## 評估機制的比較（皆訓練 1000 局，兩個 seed 平均）

| 設定 | 平均分數 / 成功率 |
|---|---|
| 沒有評估（用最後一局的模型） | 180.9 / 60% |
| 每次評估 10 局 | 266.9 / 94% |
| 每次評估 50 局（最終版） | 271.5 / 99.5% |

沒有評估時，失敗的局大多在 0～70 分，飛船懸停不降落、耗到 1000 步上限；加入評估並保留最佳模型後，成功率提升到 99.5%。

## 執行

```bash
pip install "gymnasium[box2d]" torch numpy matplotlib
```

在 notebook 中把 `train(episodes=1000, seed=0)` 的 seed 改成 0 或 1 即可重現對應結果。
