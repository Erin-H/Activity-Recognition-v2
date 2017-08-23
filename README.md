# Activity-Recognition-v2
Multi-classification problem, BP Neural Network, matlab
#
### 1.核心代码1：自行实现

* 预处理过程：
```matlab
Fs = 100;                                           % 采样率
Wc=2*10/Fs;                                         % 截止频率                              [b,a]=butter(5,Wc,'low');                        % 5阶巴特沃斯低通滤波器
temp=filter(b,a,ac.sensor_readings);
```

* 训练过程：
```matlab
%% 网络参数的初始化
numInputs = 57;                                   % 输入信号
numHidden = 90;                                   % 隐层神经元数
numOutputs = 12;                                  % 输出层神经元数
w1 = rands(numHidden,numInputs);               % 输入层、隐层间的权值矩阵
b1 = rands(numHidden,1);                        % 输入层、隐层间的偏置值
w2 = rands(numOutputs,numHidden);             %隐层、输出层间的权值矩阵
b2 = rands(numOutputs,1);                       %隐层、输出层间的偏置值
 
lr=0.05;                                           % 学习率
epoch=1500;                                       % 迭代次数

%% 训练阶段
for i=1:epoch
    for j=1:20
        %隐层输出(1*h)sigmoid
        for k=1:numHidden
            hi(k)=w1(k,:)*traindata(j,:)'+b1(k);
            ho(k)=1/(1+exp(-hi(k)));
        end
        %输出层输出(1*o) purelin
        yi=w2*ho'+b2;
        yo=yi;
        %计算输出层神经元的梯度项(1*o)
        gj=train_output(j,:)-yo';
        %输出层误差修正项
        dw2=gj'*ho;
        db2=gj';
        %计算隐藏层神经元的梯度项(1*h)
        for k=1:numHidden
            sum(k)=gj*w2(:,k);
        end
        eh=ho.*(1-ho);
        %隐藏层误差修正项
        for m=1:1:numInputs
            for n=1:1:numHidden
                dw1(m,n)=eh(n)*traindata(j,m)*sum(n);
                db1(n)=eh(n)*sum(n);
            end
        end 
        
        %更新权值和偏置值
        w1=w1+lr*dw1';
        b1=b1+lr*db1';
        w2=w2+lr*dw2;
        b2=b2+lr*db2;
    end
end
```

* 测试过程：
```matlab
%% 测试阶段
py=[];                                      % 实际预测输出
for i=1:s2
    %隐层输出
    for k=1:numHidden
        pi(k)=w1(k,:)*testdata(i,:)'+b1(k);
        po(k)=1/(1+exp(-pi(k)));
    end
    %输出层输出
    py=[py,w2*po'+b2];
end
```

### 2.核心代码2：借助matlab神经网络工具实现
```matlab
%创建神经网络
net = newff( minmax(traindata') , [100 12] , { 'logsig' 'purelin' } , 'traingdx' ) ; 
 
%设置训练参数
net.trainparam.show = 100 ;
net.trainparam.epochs = 1500 ;
net.trainparam.goal = 0.01 ;
net.trainParam.lr = 0.01 ;
net.divideFcn = '';

%开始训练
net = train( net, traindata' , train_output' ) ;

%仿真，预测输出结果
Y = sim( net , testdata' ) ;
```
