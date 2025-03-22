# 「コード翻訳言語モデル」の開発パイプラインの実装手順
本リポジトリは「コード翻訳言語モデル」の開発パイプラインを実装するための一連の流れ（データセット作成からSFT、アラインメントまで）を手順化しています。  
尚、パイプラインの実装に関しては下記、ボタンを押下してご確認ください。  
※データセット等は適宜変更ください。

↓こちらのボタンをクリック（コード翻訳パイプライン）  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazukitakayamas/llm-code-translation-tasks/blob/main/BELU-score-vllm-inference.ipynb)
<br>

### ■HuggingFaceに公開済みモデルはこちら  

[🤗 Access from HuggingFace SFT model](https://huggingface.co/kazuyamaa/gemma-2-2b-sft-merged)
<br>

[🤗 Access from HuggingFace SFT model](kazuyamaa/gemma-2-2b-code-translate-dpo-merged)  
<br>

### ■対象タスクと評価指標（前提）
[CodeTransOcean](https://github.com/WeixiangYAN/CodeTransOcean)のtest splitに対して、BELUスコアで評価を行う。  
<br>
<br>

## 1. データセット作成手順について

今回使用するのは一般に公開済みのデータセットと合成データセットになります。
<br>

### ■合成データについて
合成データの作成コードについてはMagpieの手法を使い、生成を行っています。  
モデルは、[codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)を使用しておりますが、500個のペアとなるデータを作成するのに8時間程かかりましたのでご注意ください。  
※GPU等の実行環境に大きく依存する点についてはご承知おきください。

合成データ生成のノートブックは[こちら](https://github.com/kazukitakayamas/llm-code-translation-tasks/blob/main/datasets/magpie-code-translate.ipynb)
<br>

### ■一般公開データについて

学習に使用した一般公開データは下記の通りです。  
これらをOpenAI Messages形式に変換し新たにMessagesキーを作りデータセットを作成しています。

また、DPOデータセットは[ziwenyd/transcoder-geeksforgeeks](https://huggingface.co/datasets/ziwenyd/transcoder-geeksforgeeks)という既に質の高い正解が用意されたデータセットがあり、今回はそれをDPO用の学習セットの元データとして採用しています。  
具体的には、元のデータセットを二つの言語のペアとなるように分類を行い、それぞれが完全な対応関係にあるものとして、翻訳先となる言語をChosenとしています。  
それに対して、SFTを行ったモデル（今回はgemma-2-2b）で出力（推論）をさせ、それをrejectedとしてデータを作成し、翻訳元をprompt、元の翻訳先をchosen、SFTモデルの出力がrejectedとなるような配置としてデータを作成しました。
<br>

※データセットは全て私のHuggingface内にあります。  
<br>

 -SFTデータセット  

　[WeixiangYan/CodeTransOcean](https://huggingface.co/datasets/kazuyamaa/multi-language-messages-01)
<br>

　[google/code_x_glue_cc_code_to_code_trans](https://huggingface.co/datasets/kazuyamaa/code-translate-google_messages)
<br>

　[google/code_x_glue_cc_code_refinement](https://huggingface.co/datasets/kazuyamaa/code_x_glue_cc_code_refinement_messages)
<br>

　[CodeTranslatorLLM/Code-Translation](https://huggingface.co/datasets/kazuyamaa/CodeTranslatorLLM-Code-Translation_messages)
<br>

 -DPOデータセット  

　[ziwenyd/transcoder-geeksforgeeks を基に作成したC++→pythonのデータセット](https://huggingface.co/datasets/kazuyamaa/cpp-to-python-rlhf-dataset-ver01)
<br>

　[ziwenyd/transcoder-geeksforgeeks を基に作成したJava→C++のデータセット](https://huggingface.co/datasets/kazuyamaa/java-to-cpp-rlhf-dataset-ver01)
<br>

　[ziwenyd/transcoder-geeksforgeeks を基に作成したJava→Pythonのデータセット](https://huggingface.co/datasets/kazuyamaa/java-to-python-rlhf-dataset-ver01)
<br>
<br>

## 2. 「SFT」&「DPO」について
<br>

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<p align="center">
    <img src="https://img.shields.io/github/license/axolotl-ai-cloud/axolotl.svg?color=blue" alt="GitHub License">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg" alt="tests">
    <a href="https://github.com/axolotl-ai-cloud/axolotl/releases"><img src="https://img.shields.io/github/release/axolotl-ai-cloud/axolotl.svg" alt="Releases"></a>
    <br/>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors"><img src="https://img.shields.io/github/contributors-anon/axolotl-ai-cloud/axolotl?color=yellow&style=flat-square" alt="contributors" style="height: 20px;"></a>
    <img src="https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl" alt="GitHub Repo stars">
    <br/>
    <a href="https://discord.com/invite/HhrNrHJPRb"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="discord" style="height: 20px;"></a>
    <a href="https://twitter.com/axolotl_ai"><img src="https://img.shields.io/twitter/follow/axolotl_ai?style=social" alt="twitter" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>
<br>

今回のSFTとDPOには[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)というライブラリを使用しました。  
実行には、あらかじめ用意したyamlの設定を変えるだけで学習が出来ます。  
また、DeepSpeedのマルチGPUにも柔軟に対応出来るのもうれしいポイントです。  

※[SFTyaml](https://github.com/kazukitakayamas/llm-code-translation-tasks/blob/main/sft/gemma-2-2b-config.yml)、[DPOyaml](https://github.com/kazukitakayamas/llm-code-translation-tasks/blob/main/dpo/gemma-2-2b-dpo.yml)  
※DPOについては、yamlとディレクトリ名を変える＋[gemma.py](https://github.com/kazukitakayamas/llm-code-translation-tasks/blob/main/dpo/gemma.py)をsrc/axolotl/prompt_strategies/dpo内に配置する。
<br>

### 環境構築
```
git clone https://github.com/axolotl-ai-cloud/axolotl
cd axolotl

apt-get update
apt-get install -y libopenmpi-dev
```
<br>

### 必要ライブラリのインストール
```
pip install -e .
pip install packaging ninja
pip install flash-attn
pip install deepspeed
pip install mpi4py
```
<br>

### HuggingfaceとWandbにログイン（アクセス権のあるトークンに設定してください）
```
huggingface-cli login --token "WRITE ME Your Token"
wandb login "WRITE ME Your Token"
```
<br>

### データの前処理の実行
```
python -m axolotl.cli.preprocess gemma-2-2b-dpo.yml --debug
```
<br>

### 学習の実行
```
accelerate launch -m axolotl.cli.train gemma-2-2b-dpo.yml --deepspeed deepspeed_configs/zero3_bf16.json
```
<br>

### LoRAアダプタのマージ
```
python -m axolotl.cli.merge_lora gemma-2-2b-dpo.yml --lora-model-dir="/workspace/data/models/gemma-2-2b-code-translate-simpo-merged"
```
<br>

### マージ済みのモデルをHuggingfaceへアップロード
```
cp /workspace/data/models/gemma-2-2b-code-translate-simpo-merged/README.md /workspace/data/models/gemma-2-2b-code-translate-simpo-merged/merged

huggingface-cli upload-large-folder Aratako/gemma-2-2b-code-translate-simpo-merged-merged --repo-type=model /workspace/data/models/gemma-2-2b-code-translate-simpo-merged/merged
```
<br>

## 3. パイプラインの実行
最後に冒頭で記載した以下のパイプラインを実行することで、今回のBELUの評価を行う事が出来ます。

↓こちらのボタンをクリック（コード翻訳パイプライン）  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazukitakayamas/llm-code-translation-tasks/blob/main/BELU-score-vllm-inference.ipynb)
