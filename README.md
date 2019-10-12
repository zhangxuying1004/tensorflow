# tensorflow tricks
## 1 我所使用的运行环境 
```
CUDA version: 9.0.176  
CUDNN version: 7.4.2  
tensorflow-gpu==1.8.0  
pyhton==3.6.9
```
## 2 keras框架   
```
from tensorflow.python.keras import Model, Sequential, layers, application
```
2.1 使用keras框架定义神经网络类   
```
class class_name(Model):
     def __init(self, *args):
          super(class_name, self).__init__()
          # 变量的定义
          # 网络层或者网络结构的定义
     def call(self, inputs, training=None, mask=None):
          # 网络的前向传播
# 调用
model = class_name(*args)
out = model(x)
```
2.2 使用keras框架定义网络层
```
# 全连接层    
Layer = layers.Dense(units=, activation=None)
# 卷积层  
Layer = layers.Conv2D(filters=2, kernel_size=, strides=, padding=, )
# 循环神经网络
Layer = layers.LSTM(units=)

# 拼接layers定义的网络层
model = Sequential(
     Layer1,
     Layer2,
     ...
)
# 调用
out = model(x)
```
2.3 使用keras框架训练模型   
```
model.compile(
    optimizer=optimizers.Adam(lr=learning_rate),
    loss=losses.categorical_crossentropy,
    metric=['accuracy']
)
model.fit(
    train_x, train_y,
    batch_size, epochs
)
model.evalute(eval_x, eval_y, batch_size)
```
2.4 输出每一层网络
```
for i in range(len(model.layers)):
    print(model.get_layer(index=i).output)

for layer in model.layers:
	print(layer.output_shape)
```
2.5 使用keras框架做迁移学习  
```
base_mode = applications.resnet50.ResNet50(weights='imagenet', include_top=False)
# 从任意中间层中抽取特征,index为层的索引
model = Model(inputs=base_model.input, outputs=base_model.get_layer(index=90).output)
out = model(x)
```

