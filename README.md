ComfyUI_DiffRhythm2 是一个基于小米 DiffRhythm2 模型的 ComfyUI 扩展节点，能够通过 文本提示词/参考歌曲+歌词 生成高质量的音乐作品。

https://github.com/user-attachments/assets/9d67a3df-893c-4ede-9364-10f8b7ca4431

## 🚀 安装方法

Windows 系统做如下配置. 

下载安装最新版 [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

添加系统环境变量 `PHONEMIZER_ESPEAK_LIBRARY`, 值是你安装的 espeak-ng 软件中 `libespeak-ng.dll` 文件的路径, 例如: `C:\Program Files\eSpeak NG\libespeak-ng.dll`.

Linux 系统下, 需要安装 `espeak-ng` 软件包. 执行如下命令安装:

`apt-get -qq -y install espeak-ng`

---

1. 进入 ComfyUI 的 `custom_nodes` 目录：
```bash
cd ComfyUI/custom_nodes
```

2. 克隆本仓库：
```bash
git clone https://github.com/billwuhao/ComfyUI_DiffRhythm2.git
```

3. 安装依赖：
```bash
cd ComfyUI_DiffRhythm2
pip install -r requirements.txt
```

4. 重启 ComfyUI

## 📋 使用说明

### 节点输入参数

#### 必需参数
- **音乐风格提示词** (`音乐风格提示词`): 描述想要的音乐风格，如"Vocal, Indieie, Pop, Synthesizer, Piano, Electric Guitar, Rock, Happy, Romantic"
- **歌词** (`歌词`): 输入歌词文本，支持结构标签（见下方说明）
- **歌曲最大长度** (`歌曲最大长度`): 设置生成歌曲的最大长度（秒），范围 10-500 秒，通常生成2~3分钟歌曲。

#### 可选参数
- **参考音乐** (`参考音乐`): 上传参考音频文件，生成相似风格的音乐
- **步数** (`步数`): 扩散模型采样步数，默认 20，范围 10-100
- **cfg** (`cfg`): 分类器自由引导强度，默认 2.0，范围 1.0-10.0
- **seed** (`seed`): 随机种子，用于重现结果
- **卸载模型** (`卸载模型`): 生成完成后是否卸载模型以释放内存

### 🏗️ 歌词结构标签

支持以下结构标签来组织歌词结构：

```
[start] - 开始标记
[end] - 结束标记
[intro] - 前奏
[verse] - 主歌
[chorus] - 副歌
[outro] - 尾奏
[inst] - 器乐部分
[solo] - 独奏部分
[bridge] - 桥段
[hook] - 钩子
[break] - 间歇
[stop] - 停止
[space] - 空间/停顿
```

#### 歌词示例
```
[start]
[intro]
[verse]
在这美丽的夜晚
星光洒满了天边
[chorus]
让我们一起歌唱
唱出心中的梦想
[verse]
微风轻拂过脸庞
带来了花香芬芳
[chorus]
让我们一起歌唱
唱出心中的梦想
[outro]
[end]
```

### 模型下载

**首次使用时会自动下载模型。**

可自己手动下载到 `ComfyUI\models\TTS\DiffRhythm` 文件夹下。

结构如下:

```
.
├─DiffRhythm2
│      config.json
│      decoder.bin
│      decoder.json
│      model.safetensors
│
├─MuQ-large-msd-iter
│      config.json
│      model.safetensors
│
├─MuQ-MuLan-large
│      config.json
│      pytorch_model.bin
│
└─xlm-roberta-base
        config.json
        model.safetensors
        sentencepiece.bpe.model
        tokenizer.json
        tokenizer_config.json
```

手动下载地址:
- https://huggingface.co/ASLP-lab/DiffRhythm2/tree/main  
- https://huggingface.co/OpenMuQ/MuQ-MuLan-large/tree/main  
- https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/main
- https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main

## 🙏 致谢

[xiaomi-research/diffrhythm2](https://github.com/xiaomi-research/diffrhythm2)
