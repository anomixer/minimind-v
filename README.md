<div align="center">

![logo](./images/logo.png)

</div>


<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind-v)
[![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind-v?style=social)](https://github.com/jingyaogong/minimind-v/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind-v?v=1)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind-v)](https://github.com/jingyaogong/minimind-v/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/jingyaogong/minimind-v/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniMindV%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d)

</div>

<div align="center">

![GitHub Trend](https://trendshift.io/api/badge/repositories/13265)

</div>


<div align="center">
  <h3>"大道至簡"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

* 此專案旨在從0開始，僅用1.3塊錢成本 + 1小時！即可訓練出26M引數的超小多模態視覺語言模型**MiniMind-V**。
* **MiniMind-V**最小版本體積僅為 GPT3 的約 $\frac{1}{7000}$，力求做到個人GPU也可快速推理甚至訓練。
* **MiniMind-V**是[MiniMind](https://github.com/jingyaogong/minimind)純語言模型的視覺能力額外拓展。
* 專案同時包含了VLM大模型的極簡結構、資料集清洗、預訓練(Pretrain)、監督微調(SFT)等全過程程式碼。
* 這不僅是一個開源VLM模型的最小實現，也是入門視覺語言模型的簡明教程。
* 希望此專案能為所有人提供一個拋磚引玉的示例，一起感受創造的樂趣！推動更廣泛AI社群的進步！

> 為防止誤解，“1小時” 基於NVIDIA 3090硬體裝置（單卡）測試`1 epoch`，“1.3塊錢” 指GPU伺服器租用成本。



<div align="center">

![minimind2-v](./images/minimind2-v.gif)

[🔗🤖線上體驗](https://www.modelscope.cn/studios/gongjy/MiniMind-V) | [🔗🎞️影片介紹](https://www.bilibili.com/video/BV1Sh1vYBEzY)

</div>

# 📌 Introduction

“用樂高拼出一架飛機，遠比坐在頭等艙裡飛行更讓人興奮！”
構建VLM正規化的多模態大模型是否真的如想象中那樣複雜？它的程式碼實現到底如何？
訓練過程究竟難不難？那麼現在，探索它們的答案，一起感受創造的樂趣吧！

> [!TIP]
> （截至2025-02-20）MiniMind-V 系列已完成了以下型號模型訓練，最小僅需26M (0.026B)，即可具備識圖和對話的能力！

| 模型 (大小)                   | 推理佔用   | release    | 
|---------------------------|--------|------------|
| MiniMind2-V (104M)        | 0.6 GB | 2025.02.20 |
| MiniMind2-Small-V (26M)   | 1.1 GB | 2025.02.20 |
| minimind-v-v1-small (27M) | 0.6 GB | 2024.10.04 |
| minimind-v-v1 (109M)      | 1.1 GB | 2024.10.04 |

### 👉**最近更新**

<details close> 
<summary> <b>2025-10-24</b> </summary>

- bug修復：模型權重不對應
- 適配[「minimind-1024更新」](https://github.com/jingyaogong/minimind)
- 程式碼重構：訓練和評估指令碼規範化
- 新增完整的斷點續訓支援

</details>

<details close> 
<summary> <b>2025-04-27</b> </summary>

- 相容性更新
- 適配[「minimind倉庫新特性」](https://github.com/jingyaogong/minimind/issues/370)
- 規範化部分程式碼

</details>

<details close> 
<summary> <b>2025-02-20</b> </summary>

- MiniMind2-V伴隨MiniMind2同步更新
- 大幅減少所有冗餘程式碼，規範程式碼格式
- 大幅精簡模型冗餘結構
- 更新資料集格式，拓展新的SFT資料集
- 比前代VLM更優秀的效果！

</details>

<details close>

<summary> <b>More...</b> </summary>

**2024-10-05**

- MiniMind-V如期而至，首次開源

</details>

# 📌 快速開始

<details style="color:rgb(128,128,128)">
<summary>分享本人的軟硬體配置（僅供參考）</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

### 第0步

```bash
# 克隆程式碼倉庫
git clone https://github.com/jingyaogong/minimind-v
```

```bash
# 下載clip模型到 ./model/vision_model 目錄下
git clone https://huggingface.co/openai/clip-vit-base-patch16
# or
git clone https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch16
```

```bash
# 下載minimind語言模型權重到 ./out 目錄下（作為訓練VLM的基座語言模型）
# HuggingFace
https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch/blob/main/llm_512.pth # or llm_768.pth
# 國內源
https://modelscope.cn/models/gongjy/MiniMind2-V-PyTorch/resolve/master/llm_512.pth # or llm_768.pth
```

## Ⅰ 測試已有模型效果

### 1.環境準備

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.下載模型

```bash
git clone https://huggingface.co/jingyaogong/MiniMind2-V
```

### 3.命令列問答

```bash
# load_from='model': 載入原生PyTorch權重, load_from='其他路徑': 載入transformers格式
python eval_vlm.py --load_from model --weight sft_vlm

# 或使用transformers格式模型
python eval_vlm.py --load_from MiniMind2-V
```

### 4.或啟動WebUI

```bash
python web_demo_vlm.py
```

## Ⅱ 從0開始自己訓練

### 1.環境準備

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<details style="color:rgb(128,128,128)">
<summary>注：提前測試Torch是否可用cuda</summary>

```bash
import torch
print(torch.cuda.is_available())
```

如果不可用，請自行去[torch_stable](https://download.pytorch.org/whl/torch_stable.html)
下載whl檔案安裝。參考[連結](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 2.資料下載

從下文提供的[資料集連結](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset)
下載所需內容並放到`./dataset`下。

<details style="color:rgb(128,128,128)">
<summary>注：資料集須知</summary>

Pretrain資料：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_data.jsonl
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_images.zip
unzip pretrain_images.zip && rm pretrain_images.zip
```

SFT資料：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_data.jsonl
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_images.zip
unzip sft_images.zip && rm sft_images.zip
```

`*.jsonl`為問答文字，`*images`為配套的圖片資料，下載完成後需要解壓影像資料。

請預留~5GB空間存放資料集，若無多餘空間存放pretrain資料，可嘗試跳過pretrain訓練步驟直接進行sft訓練。

</details>

### 3.開始訓練

**3.1 預訓練（學影像描述）**

```bash
# 基礎訓練命令（從LLM權重開始，僅訓練vision_proj）
python train_pretrain_vlm.py --epochs 4 --from_weight llm
```

> 執行預訓練，得到 `pretrain_vlm_*.pth` 作為預訓練的輸出權重（其中*為模型的dimension，預設為512）


**3.2 監督微調（學看圖對話方式）**

```bash
# 基礎訓練命令（從預訓練權重開始，全引數微調）
python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm
```

> 執行監督微調，得到 `sft_vlm_*.pth` 作為指令微調的輸出權重

<details style="color:rgb(128,128,128)">
<summary>注：訓練須知</summary>

**訓練特性：**
- 支援斷點續訓：新增`--from_resume 1`引數可從上次中斷處繼續訓練
- 支援GPU數量變化：續訓時GPU數量改變會自動轉換step
- 原子性儲存：使用臨時檔案+替換機制，防止儲存過程中斷導致權重損壞
- 每次儲存同時生成`out/**.pth`（模型權重）和`checkpoints/**_resume.pth`（訓練狀態）檔案

```bash
# 訓練中斷後，使用相同命令並新增 --from_resume 1
python train_sft_vlm.py --epochs 4 --from_resume 1
```

**引數說明：**
- `--from_weight`: 基礎權重名稱（llm, pretrain_vlm, none等）
- `--save_weight`: 儲存權重的字首名
- `--from_resume`: 是否續訓（0=從頭開始，1=從檢查點繼續）
- `--freeze_llm`: 是否凍結LLM引數（僅pretrain使用）
- 更多可直接參考程式碼

</details>


---

### 4.測試模型效果

確保需要測試的模型`*.pth`檔案位於`./out/`目錄下。
也可以直接去[此處](https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch)下載使用我訓練的`*.pth`檔案。

```bash
# 測試SFT模型（預設）
python eval_vlm.py --weight sft_vlm

# 測試Pretrain模型
python eval_vlm.py --weight pretrain_vlm
```

---

> [!TIP]
> 訓練指令碼均為Pytorch原生框架，均支援多卡加速，假設你的裝置有N (N＞1) 張顯示卡：

單機N卡啟動訓練方式 (DDP, 支援多機多卡叢集)

```bash
torchrun --nproc_per_node N train_xxx.py
```

<details style="color:rgb(128,128,128)">
<summary>注：其它須知</summary>

<del>
單機N卡啟動訓練 (DeepSpeed)

```bash
deepspeed --master_port 29500 --num_gpus=N train_xxx.py
```
</del>

可根據需要開啟wandb記錄訓練過程

```bash
# 需要登入: wandb login
torchrun --nproc_per_node N train_xxx.py --use_wandb
# and
python train_xxx.py --use_wandb
```

透過新增`--use_wandb`引數，可以記錄訓練過程，訓練完成後，可以在wandb網站上檢視訓練過程。透過修改`wandb_project`
和`wandb_run_name`引數，可以指定專案名稱和執行名稱。

【注】：25年6月後，國內網路環境無法直連WandB，MiniMind專案預設轉為使用[SwanLab](https://swanlab.cn/)作為訓練視覺化工具（完全相容WandB API），即`import wandb`改為`import swanlab as wandb`即可，其他均無需改動。

</details>

# 📌 VLM Detail

MiniMind-V (VLM)的基座語言模型MiniMind (LLM)來自孿生專案[minimind](https://github.com/jingyaogong/minimind)，
具體的模型結構、訓練細節、原理、測試效果等均可移步[minimind](https://github.com/jingyaogong/minimind)專案查閱。
此處為減少冗餘，省略討論LLM的相關部分，預設您已對MiniMind (LLM)的細節有基本的瞭解。

> 即使您不太瞭解LLM的細節，也可參考“快速開始”流程訓練一個MiniMind-V，
> 這並不受到影響，倉庫致力於最低成本的開箱即用！

MiniMind-V的結構僅增加Visual Encoder和特徵投影兩個子模組，增加模態混合分支，以支援多種模態資訊的輸入：
![LLM-structure](./images/VLM-structure.png)
![LLM-structure](./images/VLM-structure-moe.png)


<details>
<summary> 【重要】一些有趣的思考 </summary>

此處不妨展開想一想兩個問題：

* 什麼叫做**L**arge **L**anguage **M**odel (LLM)？
* 什麼叫做多模態模型？

[這篇文章](https://www.jiqizhixin.com/articles/2024-09-15-3)完美吻合本人的想法：
大語言模型（LLM）名字雖然帶有語言二字，但它們其實與語言關係不大，這只是歷史問題，更確切的名字應該是自迴歸 Transformer
或者其他。LLM 更多是一種統計建模的通用技術，它們主要透過自迴歸 Transformer 來模擬 token 流，而這些 token
可以代表文字、圖片、音訊、動作選擇、甚至是分子等任何東西。
因此，只要能將問題轉化為模擬一系列離散 token 的流程，理論上都可以應用 LLM 來解決。
實際上，隨著大型語言模型技術棧的日益成熟，我們可能會看到越來越多的問題被納入這種建模範式。也就是說，問題固定在使用 LLM
進行『下一個 token 的預測』，只是每個領域中 token 的用途和含義有所不同。

[ZJU-LiXi老師](https://person.zju.edu.cn/xilics#694283)同樣談及過類似觀點（原話大意如下）：
文字、影片、語音、動作等在人類看來屬於「多模態」訊號，但所謂的「模態」其實只是人類在資訊儲存方式上的一種分類概念。
就像`.txt`和`.png`檔案，雖然在視覺呈現和高階表現形式上有所不同，但它們本質上並沒有根本區別。
之所以出現「多模態」這個概念，僅僅是因為人類在不同的感知層面上對這些訊號的分類需求。
然而，對於機器來說，無論訊號來自何種「模態」，最終它們都只是以一串二進位制的「單模態」數字序列來呈現。
機器並不會區分這些訊號的模態來源，而只是處理和分析這些序列背後所承載的資訊內容。

個人認為**G**enerative **P**retrained **T**ransformer (GPT) 比 **L**arge **L**anguage **M**odel (LLM)更為貼切，
因此本人表達上更習慣用"GPT"去代表LLM/VLM/類GPT架構的系列模型，而非為了蹭OpenAI的熱度。

至此，我們可以用一句話總結GPT的所作所為：

GPT模型根據現有token預測輸出下一個下下一個下下下一個token ...，直到模型輸出結束符；此處的"token"其實並不需要一定是文字！

```text
> 對於LLM模型，如果需要理解"圖片"，我們只要把"圖片"作為對一種特殊的從來沒見過的"外國語言"，透過"外語詞典"翻譯後即可作為特殊的語言輸入LLM
> 對於LLM模型，如果需要理解"音訊"，我們只要把"音訊"作為對一種特殊的從來沒見過的"外國語言"，透過"外語詞典"翻譯後即可作為特殊的語言輸入LLM
> ...
```

<u>**為了得到MiniMind-V，我們只需要完成這2件事即可：**</u>

1. 藉助擅長翻譯圖片的 **"外語詞典"** ，把圖片從 **"外國語言"** 翻譯為模型便於理解的 **"LLM語言"**
2. 訓練微調LLM，使其和 **"外語詞典"** 度過磨合期，從而更好的理解圖片

"外語詞典" 稱之為Visual Encoder模型。
和LlaVA、Qwen-VL等視覺語言模型類似，MiniMind-V同樣選用開源Clip系列模型作為Visual Encoder。
具體使用[clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)，
一種基於 ViT-B/16 架構的經典Visual Encoder用於描述影像文字資訊。
輸入的影像尺寸為224x224，因為劃分的Patch是16×16，所以會產生14*14=196個token作為encoder編碼層的輸入，
最終產生1×768維的嵌入向量用於和文字對計算誤差。
我們並不需要最終嵌入表示，因此只取encoder層的輸出，也就是VIT核心主幹的輸出特徵即可。
它拿到前一層維度196×768大小的特徵，我們把它作為196個visual token輸入MiniMind-V。
與LLM的結合在獲取影像encoder特徵後，一方面需要把768維度的visual token對齊到LLM的文字token，
另一方面，要將影像特徵對映到與文字embedding相同的空間，即文字token和原生的視覺token需要磨合並不能直接地一視同仁，
可以稱之為跨模態的特徵對齊。
[LlaVA-1](https://arxiv.org/pdf/2304.08485)使用簡單的無偏線性變換完成了這一操作，效果很不錯，MiniMind-V同樣如此。

![llava-structure](./images/llava-structure.png)

至此，MiniMind-V的內部結構變化已經呈現完畢。

</details>


---

下面，我們簡單討論MiniMind-V的外部輸入輸出的變化。

VLM的輸入依然是一段文字，其中包含特殊的`<image>`佔位符。
在計算文字嵌入後，可以將影像編碼器生成的向量投影到該佔位符對應的嵌入部分，替換掉原先的佔位符embedding。
例如：

```text
<image>\n這個影像中有什麼內容？
```

在`minimind-v`中，使用196個字元組成的 `@@@...@@@`
佔位符代替影像，之所以是196個字元，前面有所提及：
任何影像都被clip模型encoder為196×768維的token，
因此`minimind-v`的prompt為：

```text
@@@......@@@\n這個圖片描述的是什麼內容？
```

計算完embedding和projection，並對影像部分token替換後整個計算過程到輸出則和LLM部分沒有任何區別。

![input](./images/minimind-v-input.png)

一次性多圖的實現方法就是透過注入多個`<image>`影像佔位符進行實現，不需要修改任何框架。

<details>
<summary> 影片理解的拓展思路 </summary>

write by [@xinyanghuang7](https://github.com/xinyanghuang7)

對於多模態大模型的影片理解能力，一個可行的思路是參考現有MiniCPM-V 2.6 進行影片理解的Python示例。
主要思想是透過提取影片關鍵幀，而後進行多圖推理。
因此，如果希望在MiniMind-V中新增影片理解能力，可以在現有多圖訓練的基礎上，參考此python指令碼中對於關鍵幀的提取方法，而後加大訓練檔案中支援圖片的數量。
所支援的MAX_NUM_FRAMES越多，所消耗的視訊記憶體越大。

```text
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu  # pip install decord

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number


def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames


video_path = "video_test.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]},
]

# Set decode params for video
params = {}
params["use_image_id"] = False
params["max_slice_nums"] = 2  # 如果cuda OOM且影片解析度大於448*448可設為1

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)
print(answer)
```

</details>

至此，`MiniMind-V`的所有細節已經呈現完畢。
`MiniMind-V`的模型子類完全繼承自`MiniMind`，
僅基於後者做**最小**變更而產生，
其核心演算法改動`< 50行`，遷移難度極低。
因此可能和`LlAVA`等模型細節可能存在區別，但思路完全統一。

# 📌 Experiment

## Ⅰ 資料集

來源：[Chinese-LLaVA-Vision](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)
包含約57萬張預訓練影像，來自CC-3M和COCO 2014；
[llava-en-zh-300k](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k)
包含300k條指令微調資料和15萬張影像。
問答內容經過翻譯，
對中文支援更友好，進一步經過整理並`resize`。

(pretrain_vlm_data.jsonl) 預訓練資料集格式：

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "提供給定影像的簡要描述。\n<image>"
    },
    {
      "role": "assistant",
      "content": "橄欖油是自由使用的健康成分。"
    }
  ],
  "image": "GCC_train_002582585.jpg"
}
```

(sft_vlm_data.jsonl) 單圖指令微調資料集格式：

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "鬧鐘的位置對睡眠質量有什麼影響？<image>"
    },
    {
      "role": "assistant",
      "content": "把數字鬧鐘放在床頭櫃..."
    }
  ],
  "image": "train-00000-of-00001_image_0_0.jpg"
}
```

(sft_vlm_data_multi.jsonl) 多圖指令微調資料集格式：

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "context: Source Image: <image> Target Image: <image> Instruction: What is the correct image edit instruction that can transfrom the source image to target image?<image>"
    },
    {
      "role": "assistant",
      "content": "take the people out of the back in the photo. Remove the two people behind the woman in the white dress and the man in the blue suit. remove people behind the couple in the centre"
    }
  ],
  "image": "0.jpg, 1.jpg"
}
```

<details>
<summary> 資料說明 </summary>

* 多圖資料集規模相對較小且為英文對話，資料集僅包含兩圖對比的場景，因此微調效果有限，這裡只提供一種參考思路。


* `jsonl`均為文字指令，`images.zip`均為配套的影像資料（下載後需要解壓）

</details>

資料集下載地址：([ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset))

## Ⅱ 訓練

> train_pretrain_vlm

預訓練從595K條資料集中學習圖片的通用知識，比如鹿是鹿，狗是狗。

> train_sft_vlm

指令微調從300K條真實對話資料集中學習對圖片提問的真實問答格式，更符合與人類的交流習慣。

> train_sft_vlm

多圖微調提供demo：鳥類對比資料集，長度為13.6k的真實問答格式。

訓練時均凍結visual encoder也就是clip模型梯度，
只訓練Projection和LLM兩部分。
預訓練中，只設置Projection和LLM的最後一層引數可學習。
指令微調中，設定Projection和LLM的全部引數可學習。

> 訓練時間和Loss走勢（僅供參考）

Pretrain [512+8] & [768+16]
![input](./images/pretrain_loss.png)

SFT [512+8] & [768+16]
![input](./images/sft_loss.png)

## Ⅲ 模型權重

(原生PyTorch`*.pth`權重檔案) 下載地址：
([ModelScope](https://www.modelscope.cn/models/gongjy/MiniMind2-V-PyTorch) | [HuggingFace](https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch))

(`Transformers`格式模型)
下載地址：
([ModelScope](https://www.modelscope.cn/profile/gongjy) | [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-v-67000833fb60b3a2e1f3597d))

> 注：Transformers版本均為單圖指令微調後的`MiniMind-V`模型

# 📌 Test

### 效果測試

#### 單圖對話

<table>
  <thead>
    <tr>
      <th>圖片</th>
      <th>MiniMind2-V</th>
      <th>MiniMind2-V-Small</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="./dataset/eval_images/城市車水馬龍-city-traffic.jpg" alt="city-traffic">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>圖中是一個繁忙的城市街道，一條長長的街道兩旁都是高樓大廈。這條街上擠滿了汽車、卡車和公共汽車，還有許多其他車輛在路上行駛。在街道上，可以看到許多汽車，有的在高速行駛，而其他的則停在街道一側。此外還有一輛公交車也停在街道的右側。街道上可以看到交通燈，表明這是一個繁忙的城市環境。</td>
      <td>圖中是一個繁忙的城市景象，有幾輛汽車和一輛卡車行駛在城市街道上。可以看到許多交通訊號燈，其中一些位於街道左側，另一些則在右側。可以看到有幾個人在街上行走，其中一些人站得離街道更近一些，而另一些則距離較遠。還有一個停車標誌位於畫面的左側，暗示著城市環境。可以看到街道上有兩輛汽車，一輛在右邊，另一輛在左邊，還有一輛在左邊。這幅影像捕捉到了都市環境中典型的一天。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/太空宇航員-Astronaut-Space.jpg" alt="astronaut">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>圖片顯示了一個宇航員的宇航員身穿宇航服，坐在一架大型太空梭上。他們似乎正在進行一次宇航員登機或下機的旅程。在宇航員的身後，有一個火箭發射架，可能是用來支撐宇航員在旅程中的任務。此外，還有一架飛機停在機庫附近，進一步表明這是一次航空展。在飛機的周圍，還有一些人，但他們看起來離飛機很近。可以看到一個人站在飛機附近，可能正在觀察或等待太空梭準備起飛。</td>
      <td>場景中，一名士兵戴著頭盔站在一架大型飛機上。這架飛機似乎是一架軍用軍用飛機，似乎正準備登上一架飛機。另一個人則站在前面，可能正在觀察飛行過程。在飛機周圍，有幾個人，其中一些站在左側，另一些則站在右側。他們似乎正在觀看飛行員的表現。此外，還有一輛卡車停在靠近左側的位置，可能是為了更具體地觀察飛行過程。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/小狗美女海邊-Dog-Woman-Sea.jpg" alt="dog-woman-sea">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>圖片中，一個女人坐在沙灘上，手裡拿著一隻白色的狗。她看起來像是個女人，坐在沙地上，看著她。一隻狗也坐在她旁邊，看起來很放鬆和舒適。海灘上散佈著其他沙灘遊客，有些人坐著，而另一些人則坐在更遠的地方。背景中可以看到一艘船，這表明這是一個受歡迎的海灘旅遊目的地。</td>
      <td>兩個人坐在海灘上，一邊懶洋洋地躺在沙灘上，另一邊則坐著。他們似乎正在享受海邊時光。海灘上有幾把椅子，其中一把靠近沙灘的左側，另一把在中間。此外，還有一隻狗躺在沙地上，為這個場景增添了一種放鬆的氣氛。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/彩虹瀑布-Rainbow-Falls.jpg" alt="rainbow-falls">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>照片捕捉到一幅美麗如畫的大自然場景，背景是高山巒崖。在水邊，一座巨大的噴泉橫跨著水面，吸引著許多遊客。水面上有幾個人，他們或站或坐在噴泉周圍，或站或坐。有些人可以看到他們在水中行走，而其他人則站在水邊。總體而言，這幅畫描繪的是一個美麗而寧靜的環境，在那裡人們可以欣賞到如畫般的美景。</td>
      <td>在一個美麗的藍色天空下，一座巨大而巨大的白色瀑布上方懸掛著一隻巨大的溼流水。這隻瀑布位於一座山上，為整個場景增添了一種迷人而又寧靜的氣氛。在這幅影像的背景中，可以看到幾艘船，其中一些靠近水邊，其他的則離得較遠。這些船隻似乎正在為風景或戶外活動做準備。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/椅子老人看書-Chair-Elderly-Reading.jpg" alt="elderly-reading">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>圖中，一個男人坐在公園的長椅上，旁邊是一把綠色椅子。他身邊有一本開啟的書，上面寫著"讀書"一句話，暗示他可能正在閱讀。公園裡有一張長椅和一張公園長椅，為周圍的環境增添了幾分生氣。在公園的周圍，有幾輛汽車和一輛卡車，表明這是一個公共區域。此外，還可以看到一個人站在公園的不同位置上，可能是等著上路或過馬路。</td>
      <td>一個穿著短褲的老人坐在公園長椅上，周圍是樹木。他似乎正在讀一本書，可能是在讀書。背景中有一座長凳，為這個場景提供了充足的座位。在背景中，可以看到一把椅子和一張餐桌，這說明這個場景可能是在一個戶外座位區，那裡有椅子供人們坐下來放鬆。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/熊貓草地-Panda-Grassland.jpg" alt="panda-grassland">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>圖中，一隻白色的棕熊坐在草地上，旁邊是一隻長著棕色斑點的大熊。這隻熊看起來很害羞或頑皮，因為它正躺在草地上休息，看上去很放鬆。</td>
      <td>在這幅影像中，一隻棕色的熊正在草地上漫步。這隻熊被放置在草地上，佔據了畫面的大部分空間。它似乎正在自然環境中行走，可能是在草地上。在背景中，有幾棵樹，為畫面增添了自然元素。一隻鳥在場景的中間附近飛翔，為畫面增添了生氣勃勃的氣氛。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/腳踏車鮮花-Bicycle-Flowers.jpg" alt="bicycle-flowers">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>圖片展示了一個漂亮的花瓶，裡面插滿了五顏六色的鮮花和花束。這些花束散落在整個花瓶中，給人一種賞心悅目的感覺。花瓶裡插著五顏六色鮮花，創造出一種令人賞心悅目的景象。這些鮮花被擺放在一張桌子上，很可能是為了展示它們的美麗而擺放的。</td>
      <td>場景中，一輛綠色和紫色相間的腳踏車停在一棟建築旁邊，它被放置在一棵大樹旁。這輛腳踏車被擺放在附近，為這個場景增添了幾分色彩。除了腳踏車外，還有一些其他的腳踏車，包括兩個位於前景中的一個和位於背景中靠近中心位置的另一個。腳踏車的存在表明它可能是停在那裡的。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/舞蹈-dance.jpg" alt="dance">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>圖片中的女人穿著一件白色連衣裙，腳上還繫著一條黑色網球裙。她正在表演一個網球比賽，很可能是在比賽中。在背景中可以看到幾把椅子，可能是為了觀眾或其他觀眾的座位安排而擺放的。此外，還有一個長凳放在場景左側，為人們提供了一個休息的地方。</td>
      <td>一名身穿白色衣服的女子站在舞臺上，手裡拿著一隻手拿著白色飛盤。她似乎正在參加一個舞臺舞會或比賽。場景中還有其他幾個人，其中一個站在舞臺左側，另一個站在右側，第三個人則站在場地右側。舞臺上有幾個觀眾，有的站著，有的坐著，還有一些站著。這看起來像是一場歡樂的節日或活動。</td>
    </tr>
  </tbody>
</table>

#### 多圖對話（效果十分有限）

<table>
  <thead>
    <tr>
      <th>圖片1</th>
      <th>圖片2</th>
      <th>512_sft_multi</th>
      <th>768_sft_multi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="./dataset/eval_multi_images/bird/0.jpg" alt="a-bird.png"></td>
      <td><img src="./dataset/eval_multi_images/bird/1.jpg" alt="a-bird.png"></td>
      <td>這幅影像顯示了一種鳥簸戮的場景：一個女人站在紅綠相間的紅綠相間的紫色鳥簸戴在她身上。女人站在紅色的鳥簸戴在她身上，而她的翻領上的那隻紅鳥則站在她身後。</td>
      <td>這兩隻鳥在同一片樹林中飛翔，有的位於畫面中心，而另一些則較小，形成了鮮明對比。這種鳥類的出現突出了其飛行能力和適應性，因為它們能夠在樹林中快速迅速移動。此外，兩隻鳥的位置不同，一個在影像的左邊，另一個在右邊，這表明它們在同一片樹林中移動得很近。這種鳥類的自然行為也有助於區分這兩種鳥類物種。</td>
    </tr>
  </tbody>
</table>

### 效果小結：

視覺訊號對於LLM視作一種特殊的外語，
因此“學習外語”的能力高低，
很大程度上取決於LLM的能力。
LLM效能越強，對應的VLM必然越強，此時效果增益會很明顯。

#### 未來值得改進的方面：

```text
> 更簡單的Projection的跨模態特徵對齊方式，相較於Cross-Attention可能處於劣勢。
> Clip模型可以嘗試更大效能更強的large系列，用更具細粒度的token表徵影像特徵，目前仍粗糙。
> 解析度不高，理論上只有224×224（minimind-v資料集為節省空間，僅設定為128×128）。
> ...
```

# 📌 Acknowledge

> [!TIP]
> 如果您覺得 `MiniMind-V`對您有所幫助，可以在 GitHub 上加一個⭐<br/>
> 水平有限難免存在未知的紕漏，歡迎所有人在Issues交流指正或提交PR改進專案<br/>
> 您的支援就是持續改進專案的動力，謝謝！

## 🤝[貢獻者](https://github.com/jingyaogong/minimind/graphs/contributors)

<a href="https://github.com/jingyaogong/minimind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jingyaogong/minimind-v" />
</a>

## 😊鳴謝

<a href="https://github.com/xinyanghuang7"><b>@xinyanghuang7</b></a>:
<a href="https://github.com/xinyanghuang7/minimind-v/tree/hxy">🔗實現了完整的多圖分支</a>

<details close> 
<summary> <b>參考連結 & 感謝以下優秀的論文或專案</b> </summary>

- 排名不分任何先後順序
- [LlaVA](https://arxiv.org/pdf/2304.08485)
- [LlaVA-VL](https://arxiv.org/pdf/2310.03744)
- [Chinese-LLaVA-Vision-Instructions](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)

</details>

## 🫶支持者

<a href="https://github.com/jingyaogong/minimind-v/stargazers">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/stars/dark/jingyaogong/minimind-v"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/stars/jingyaogong/minimind-v"/>
      <img alt="github contribution grid snake animation" src="https://reporoster.com/stars/jingyaogong/minimind-v"/>
    </picture>
</a>

<a href="https://github.com/jingyaogong/minimind-v/network/members">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/forks/dark/jingyaogong/minimind-v"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/forks/jingyaogong/minimind-v"/>
      <img alt="github contribution grid snake animation" src="https://reporoster.com/forks/jingyaogong/minimind-v"/>
    </picture>
</a>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jingyaogong/minimind-v&type=Date"/>
</picture>

# 🎓 Citation

If you find MiniMind-V helpful in your research or work, please cite:

```bibtex
@misc{minimind,
  title={MiniMind-V: Train a Tiny VLM from scratch},
  author={Jingyao Gong},
  year={2024},
  howpublished={https://github.com/jingyaogong/minimind-v}
}
```

# License

This repository is licensed under the [Apache-2.0 License](LICENSE).
