# Code for Transformer to Transformer with DBS for paraphrase generation


## Introduction
这是一个复述生成的模型代码,已封装成类似 bert-as-service的服形式，如果需要自己训练可以看下这里的代码，进行适当的修改/server/bert_serving/server/page


## 模型结构图
模型结构图
![image](https://github.com/xuhaiming1996/transformer-with-diverse-beamsearch/blob/master/model.png)

可以运行下面命令开启服务

```shell

MODEL_DIR="模型路径"
page-serving-start \
--path=$MODEL_DIR/model_avg.pt \ $MODEL_DIR
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
################
page-serving-start --path=$MODEL_DIR/model_avg.pt  $MODEL_DIR --beam 10  --source-lang=src  --target-lang=tgt --tokenizer=moses --bpe=subword_nmt --bpe-codes=$MODEL_DIR/code --nbest=40 --moses-source-lang="zh" --moses-target-lang="zh" --diverse-beam-groups=10  --diverse-beam-strength=0.9 -http_port=8000 -num_worker=1





## 开始服务 这里的路径根据自己的位置改




