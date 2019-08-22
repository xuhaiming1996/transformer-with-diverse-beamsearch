# Code for Transformer to Transformer with DBS for paraphrase generation


## Introduction
这是一个复述生成的模型代码,已封装成类似 [bert-as-service](https://github.com/hanxiao/bert-as-service)的服务形式
支持python api 和http 访问


## 模型结构图
模型结构图
![image](https://gitdojo.gz.cvte.cn/xuhaiming/btla/blob/master/model.png)

可以运行下面命令开启服务

```shell
## 这里的模型路径根据自己的位置改
MODEL_DIR=/container_data/fairseq/checkpoints/transformer_for_page
page-serving-start \
--path=$MODEL_DIR/model_avg.pt \
--beam 10  \
--source-lang=src \
--target-lang=tgt\
--tokenizer=moses \
--bpe=subword_nmt \
--bpe-codes=$MODEL_DIR/code \
--nbest=40 \
--moses-source-lang="zh" \
--moses-target-lang="zh" \
--diverse-beam-groups=10 \
 --diverse-beam-strength=0.9 \
 -http_port=8000 \
 -num_worker=1

```










