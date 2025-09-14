# **Everyones_text2image 穴埋めノートブック・チュートリアル**

これは、「画像生成AIが好き！」「自分でも作ってみたい！」という人のための完全ガイドです。  
「バナナ風の人が車を運転している」と打てばそれっぽい画像を返してくれる。  
それが実は、ただの足し算掛け算と少しの非線形の積み重ねで動いていると知ったとき、
ワクワクして夜も眠れなくなる――  
ようこそ、ディープラーニングの世界へ。もう元には戻れません。  
このガイドは、毎日南北線のぎゅうぎゅうの席で、通学中に書きました。  
学生さんたちから「できた！」というコメントが届くたび、
あの狭い席で過ごした時間に大きな意味があったんだなぁと感じています。

---

さあ、一緒にDiffusionモデルを作ろう！😎  
このチュートリアルでは、わかりやすい解説と**100問以上の穴埋め問題🫨**を用意しています。  
「ちょっとLLM作りたいからやってみたい」――そんな人が本気で力をつけられる内容です。  
必要な予備知識があれば、12〜18時間で修了可能！  
すべてGoogle Colab上で動かせます。  
このチュートリアルは[Mark Liu先生のtxt2img](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)をベースにしています。  
この場を借りて、心から感謝します。

---
## 目次

| チャプター  | 推定所要時間 | ノートブック  |
|---|---|---|
| Chapter 00: Start Tutorial      | 1〜2時間 | [![Colabで開く](https://colab.research.google.com/github/kondotaichi/everyonesTextToImage/blob/main/notebooks/todo/chapter01_transformer.ipynb) |
| Chapter 01: Transformer      | 1〜2時間 | [![Colabで開く]() |
| Chapter 02: VisionTransformer     | 1〜2時間 | [![Colabで開く]() |
| Chapter 03: エンコーダ・デコーダによる画像キャプション   | 1〜2時間 | [![Colabで開く]() |
| Chapter 04: 無条件diffusion  | 1〜2時間 | [![Colabで開く]() |
| Chapter 05: 条件つきdiffusion とCFG  | 1〜2時間 | [![Colabで開く]() |
| Chapter 06: 高品質・高速化  | 1〜2時間 | [![Colabで開く]() |
| Chapter 07: CLIP（画像・テキストの対照学習）  | 1〜2時間 | [![Colabで開く]() |
| Chapter 08: Latent Diffusion 潜在空間を利用したdiffusion  | 1〜2時間 | [![Colabで開く]() |
| Chapter 08: Stable Diffusion  | 1〜2時間 | [![Colabで開く]() |

###関連図
## Chapters dependency graph (Ch2–10)

```mermaid
%%{init: {'theme':'neutral','themeVariables':{'fontSize':'14px'}}}%%
graph TD
  %% -------- Groups --------
  subgraph Foundations[Foundations_基礎]
    ch1[Ch1 Transformer 基礎]
    ch2[Ch2 Vision Transformer]
    ch3[Ch3 画像キャプション<br/>Enc-Dec 条件生成]
    ch7[Ch7 CLIP_画像×テキスト]
  end

  subgraph DiffusionCore[Diffusion Core_中核]
    ch4[Ch4 無条件 Diffusion_DDPM]
    ch5[Ch5 条件付き Diffusion + CFG]
    ch6[Ch6 画質・高速化 DDIM/EMAなど]
  end

  subgraph Latent&SD[Latent & Integration]
    ch8[Ch8 Latent Diffusion_VAE + 潜在U-Net]
    ch9[Ch9 Stable Diffusion_統合]
  end

  %% -------- Strong dependencies (solid) --------
  ch1 --> ch2
  ch1 --> ch3
  ch1 --> ch7

  ch4 --> ch5
  ch4 --> ch6
  ch5 --> ch6

  ch4 --> ch8
  ch6 --> ch9
  ch7 --> ch5
  ch7 --> ch9
  ch8 --> ch9

  %% -------- Helpful but optional (dashed) --------
  ch2 -. 画像エンコーダ/注意ブロック理解 .-> ch3
  ch3 -. 条件生成の考え方 .-> ch5
  ch4 -. Visionの設計感覚 .-> ch6
  ch5 -. 条件付きの流儀 .-> ch8

  %% -------- Legend --------
  subgraph Legend[凡例 / Legend]
    l1a(( )) -- 強い依存（実線） --> l1b(( ))
    l2a(( )) -. 補強（点線） .-> l2b(( ))
  end

  %% -------- Styling --------
  classDef core fill:#e6f7ff,stroke:#246,stroke-width:1.2px;
  classDef base fill:#f8f8f8,stroke:#999,stroke-width:1px,stroke-dasharray: 3 2;
  class ch4,ch5,ch6,ch8,ch9 core;
  class ch1,ch2,ch3,ch7 base;
  style Legend fill:#fffaf0,stroke:#ccc,stroke-dasharray: 3 2;
```


## **デモ出力例**

---

## **Tensor Map（テンソル全体図）**

---

## **前提知識・スキル**

**理解してほしいこと**  
- 行列の掛け算と足し算  
- 平均値と分散  
- ResNetの残差接続（residual connection）  
- Word2Vectorの仕組み  

**ほぼ必須の経験**  
- PyTorchでMNISTの数字認識モデルを作ったことがある  
- 上のようなモデルでクロスエントロピーを損失関数に使ったことがある  
- ミニバッチ処理やバッチ正規化（batch normalization）を実装したことがある  
- オートエンコーダやVAEを作ったことがある  

---

PyTorchやプログラミングが初めての人向けにも、前提条件を満たすための2025年8月30日までに楽しい入門チュートリアルを準備中です！

---

## **モデルについて**

このチュートリアルで使うのは、1文字＝1トークンという超シンプルな「バイグラムモデル」です。  
内部構造もかなり単純になっています。  
学習データセットはShakespeare（シェイクスピア）のテキスト。とても古いので著作権フリーです。

本物のGPT-2と比べると、ものすごく基本的な内容ですが、「基礎から学んで本物を目指す」にはこれがベスト。  
16GBメモリのPCなら、たった2〜4分のCPU学習だけでシェイクスピアっぽい文章が出てきます！  
きっと感動するはず！

---

## **開発環境について**

セットアップの手間を減らすため、サンプルはすべてGoogle Colab上で動かしてみてください。

ただしGoogle Colabでは、チェックボックスのマークが保存されません。  
進捗管理をしたい人や、30分ごとなど「ちょっとずつ進めたい」人には、VS Codeがおすすめです。  
その場合は、このリポジトリをforkして、自分のPCにcloneしてください。


Python 3.12 & PyTorch 2.6.0がベストですが、たいていは他のバージョンでも動きます。  
普通は、今インストールしているPyTorchでOK！  
もし動かない場合は `requirements.txt` で仮想環境を作ると良いです。  
Docker Desktopを使っている場合は、同梱の`Dockerfile`やDev Container拡張でさらに安定した環境を作れます。

---

## **Project EveryonesAIについて**

![EveryonesAI Logo](assets/EveryonesAI_logo.png)  
![EveryonesAI Goal](assets/EveryonesAI_goal.png)  
![EveryonesAI Idea](assets/EveryonesAI_idea.png)  
![EveryonesAI Prerequites](assets/EveryonesAI_prerequites.png)  
![EveryonesAI Plan](assets/EveryonesAI_plan.png)