## Yolact

### 模型测试

- 训练 [coco val5k](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolact/yolact_train_loss.png" width="1000" height="500"/> 

- 检测/分割效果

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolact/yolact_train_1.png" width="350" height="230"/> (<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolact/yolact_train_2.png" width="350" height="230"/>)

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolact/yolact_train_3.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolact/yolact_train_4.png" width="350" height="230"/>


### Get start

1. 训练
```python
python3 train.py
```

2. tensorboard
```python
tensorboard --host 0.0.0.0 --logdir ./logs/ --port 8053 --samples_per_plugin=images=40
```    

3. 查看
```python
http://127.0.0.1:8053
```    

4. 测试, 修改`detect.py`里面`input_image`和`model_path`
```python
python3 detect.py
```

5. 评估验证
```python
python3 val.py
```

6. 导出`TFLite`格式
```python
cd ./data
python3 ./h5_to_tfite.py
```

7. 导出`ONNX`格式
```python
cd ./data
python3 ./h5_to_onnx.py
```