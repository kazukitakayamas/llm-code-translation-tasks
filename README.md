# llm-code-translation-tasks
　本リポジトリは「コード翻訳言語モデル」の開発パイプラインを実装するための一連の流れ（データセット作成からSFT、アラインメントまで）を手順化しています。  
　尚、パイプラインの実装に関しては下記、ボタンを押下して実行ください。  
　※データセット等は適宜変更ください。

↓こちらのボタンをクリック（コード翻訳パイプライン）  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazukitakayamas/llm-code-translation-tasks/blob/main/BELU-score-vllm-inference.ipynb)
<br>
<br>

## 1. データセット作成手順について

今回使用するのは一般に公開済みのデータセットと合成データセットになります。
<br>

### 合成データについて
合成データの作成コードについてはMagpieの手法を使い、生成を行っています。  
モデルは、[codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)を使用しておりますが、500個のペアとなるデータを作成するのに8時間程かかりましたのでご注意ください。  
※GPU等の実行環境に大きく依存する点についてはご承知おきください。

合成データ生成のノートブックは[こちら](https://github.com/kazukitakayamas/llm-code-translation-tasks/blob/main/datasets/magpie-code-translate.ipynb)
<br>

### 一般公開データについて

学習に使用した一般公開データは下記の通りです。  
これらをOpenAI Messages形式に変換し新たにMessagesキーを作りデータセットを作成しています。

また、DPOデータセットは[ziwenyd/transcoder-geeksforgeeks](https://huggingface.co/datasets/ziwenyd/transcoder-geeksforgeeks)という既に質の高い正解が用意されたデータセットがあり、今回はそれをDPO用の学習セットの元データとして採用しています。  
具体的には、元のデータセットを二つの言語のペアとなるように分類を行い、それぞれが完全な対応関係にあるものとして、翻訳先となる言語をChosenとしています。  
それに対して、SFTを行ったモデル（今回はgemma-2-2b）で出力（推論）をさせ、それをrejectedとしてデータを作成し、翻訳元をprompt、元の翻訳先をchosen、SFTモデルの出力がrejectedとなるような配置としてデータを作成しました。
<br>

※データセットは全て私のHuggingface内にあります。  

 -SFTデータセット
　*[WeixiangYan/CodeTransOcean](https://huggingface.co/datasets/kazuyamaa/multi-language-messages-01)
　*[google/code_x_glue_cc_code_to_code_trans](https://huggingface.co/datasets/kazuyamaa/code-translate-google_messages)
　*[google/code_x_glue_cc_code_refinement](https://huggingface.co/datasets/kazuyamaa/code_x_glue_cc_code_refinement_messages)
　*[CodeTranslatorLLM/Code-Translation](https://huggingface.co/datasets/kazuyamaa/CodeTranslatorLLM-Code-Translation_messages)

 -DPOデータセット
　*[ziwenyd/transcoder-geeksforgeeks を基に作成したC++→pythonのデータセット](https://huggingface.co/datasets/kazuyamaa/cpp-to-python-rlhf-dataset-ver01)
　*[ziwenyd/transcoder-geeksforgeeks を基に作成したJava→C++のデータセット](https://huggingface.co/datasets/kazuyamaa/java-to-cpp-rlhf-dataset-ver01)
　*[ziwenyd/transcoder-geeksforgeeks を基に作成したJava→Pythonのデータセット](https://huggingface.co/datasets/kazuyamaa/java-to-python-rlhf-dataset-ver01)
<br>
<br>

## 2. 「SFT」&「DPO」について

今回のSFTには[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)というライブラリを使用しました。  
実行には、あらかじめ用意した[yaml](dpo/gemma-2-2b-dpo.yml)の設定を変えるだけで簡単にSFTが出来ます。  
※DPOについては、yamlとディレクトリ名を変える＋[gemma.py](https://github.com/kazukitakayamas/llm-code-translation-tasks/blob/main/dpo/gemma.py)をsrc/axolotl/prompt_strategies/dpo内に配置する。
<br>

### 環境構築
```
git clone https://github.com/axolotl-ai-cloud/axolotl
cd axolotl

apt-get update
apt-get install -y libopenmpi-dev
```

### 必要ライブラリのインストール
```
pip install -e .
pip install packaging ninja
pip install flash-attn
pip install deepspeed
pip install mpi4py
```

### HuggingfaceとWandbにログイン（アクセス権のあるトークンに設定ください）
```
huggingface-cli login --token "WRITE ME Your Token"
wandb login "WRITE ME Your Token"
```

### データの前処理の実行
```
python -m axolotl.cli.preprocess gemma-2-2b-dpo.yml --debug
```

### 学習の実行
```
accelerate launch -m axolotl.cli.train gemma-2-2b-dpo.yml --deepspeed deepspeed_configs/zero3_bf16.json
```

### LoRAアダプタのマージ
```
python -m axolotl.cli.merge_lora gemma-2-2b-dpo.yml --lora-model-dir="/workspace/data/models/gemma-2-2b-code-translate-simpo-merged"
```

### マージ済みのモデルをHuggingfaceへアップロード
```
cp /workspace/data/models/gemma-2-2b-code-translate-simpo-merged/README.md /workspace/data/models/gemma-2-2b-code-translate-simpo-merged/merged

huggingface-cli upload-large-folder Aratako/gemma-2-2b-code-translate-simpo-merged-merged --repo-type=model /workspace/data/models/gemma-2-2b-code-translate-simpo-merged/merged
```
<br>
<br>

## 3. パイプラインの実行
最後に冒頭で記載した以下のパイプラインを実行することで、今回のBELUの評価を行う事が出来ます。

↓こちらのボタンをクリック（コード翻訳パイプライン）  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazukitakayamas/llm-code-translation-tasks/blob/main/BELU-score-vllm-inference.ipynb)
