SQLova+Baseline
======
这个代码主要实现了SQLova和Baseline的集成功能，在预测cond的val部分时，用贪心算法根据数据库中的内容和问题的关联度预测val。
------
#### 模型依赖
- `python3.6` or higher。
- `PyTorch 0.4.0` or higher。
- `CUDA 9.0`
- Python libraries: `babel, matplotlib, defusedxml, tqdm`
- 第三方依赖包
    - `pip install pytorch torchvision -c pytorch`
    - `pip install -c conda-forge records==0.5.2`
    - `pip install babel` 
    - `pip install matplotlib`
    - `pip install defusedxml`
    - `pip install tqdm`
    - `pip install cn2an`
- 模型在 Ubuntu 16.04.4 LTS 和 1080Ti GPU运行，大约12小时候趋向收敛，越长的训练时间效果越好。
### 注：所有命令在code文件夹下运行

### 训练模型
1. 将文件进行预处理,进行数据增强
```
  python3 DataAugment.py 
  数据将覆盖原数据
```
  
2. 进入code文件夹进行运行，大概10轮后收敛
```
  python3 train.py --seed 1 --bS 2 --accumulate_gradients 16 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 285 
```
### 模型预测
1. 用SQLova评估模型
```
 python3 deveval.py --seed 1 --bS 2 --accumulate_gradients 16 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 285  
```
2. 用Baseline评估模型
```
python3 predict.py --model_file ../model/model_best.pt --bert_model_file ../model/model_bert_best.pt --bert_path ../model/ --data_path ../data/val/ --split val --result_path ../submit

sh ./start_test.sh 0 val 1 ../submit/i.json
```
#### 生成结果提交文件
```
python3 predict.py --model_file ../model/model_best.pt --bert_model_file ../model/model_bert_best.pt --bert_path ../model/ --data_path ../data/test/ --split test --result_path ../submit

sh ./start_test.sh 0 test 1 ../submit/ans.json

会在submit里生成ans.json用于提交
```

