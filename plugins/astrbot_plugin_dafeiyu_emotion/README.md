# 大肥鱼情绪版

这是给现成 AstrBot / 大肥鱼加的插件。普通版不看情绪。发「情绪版」之后，每句话用本机 MacBERT 判断，回复仍走你在 AstrBot 里选中的那个模型。

## 放到 AstrBot

把本目录复制为：

`AstrBot/data/plugins/astrbot_plugin_dafeiyu_emotion`

在插件页重载。WebUI 或 QQ 里发送：

- `情绪版`：打开
- `普通版`：关掉

普通版里出现感受词时，会问一次要不要换成情绪版。词从大连理工情感词汇本体的乐、哀、怒、惧、烦、惊里挑了聊天会说的那些，例如开心、难过、生气、害怕、焦虑、压力、烦躁、委屈。回「情绪版」才换，回「不用」就继续普通聊天。整张表在配置 `switch_keywords` 里，可以改。

电脑上还没有 `best.pt` 时，配置里的分类器保持 `debug`。这时只是用关键词把纸条走通，方便看模型会不会按纸条说话，不能当成识别结果。

## 把 MacBERT 下载到本机

基座可以从 Hugging Face 或国内镜像下到本地，大约几百 KB 的分词器和配置：

```bash
python download_macbert.py --dest models/chinese_macbert_base
```

情绪分类权重 `best.pt` 是训练出来的，公开页面没有。从训练机器复制这一份：

`experiments/smp_text_emotion_v1/runs/matched_v1/macbert/seed42/best.pt`

放到同一个 `models/chinese_macbert_base/best.pt`。脚本会核对 sha256：

`38b9300f32048d87696146e2d72c8b3cad0f4e02597aec82abfc09bf84d06bbd`

然后在插件配置里填写：

- `classifier` = `macbert`
- `macbert_base_dir` = 上面的目录
- `macbert_checkpoint` = `best.pt` 的完整路径
- `macbert_device` = `cpu`

AstrBot 使用的 Python 需要有 `torch` 和 `transformers`。

## 本机先看纸条

```bash
python debug_cli.py
```

MacBERT 只给六类粗分类。过线之后，程序只把粗分类、第二名和原话交给当前模型，由它判断这一轮该做什么。程序不指定听、复述或拆步骤。
