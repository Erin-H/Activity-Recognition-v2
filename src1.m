%% --------------------------------------------------
% 基于BP神经网络的人体运动状态识别
% ①.预处理:巴特沃斯低通滤波器降噪处理
% ②.特征提取和选择：57维特征
% ③.分类识别：BP神经网络分类器（隐藏层+输出层）
%  神经元数：输入层（57）隐藏层（90）输出层（12）
%  学习规则：梯度下降法
%  传输函数：隐藏层（sigmoid函数）、输出层(线性函数)
%  训练集：13名受试者、12个动作的4个时刻共780个采集样本
%  测试集：其余60个样本
% ----------------------------------------------------------- 

clear;clc;
data1=[];
tic;
%% 预处理+特征提取
Fs = 100;                                                                  %采样率
for i=1:14
  for j=1:12
    for k=1:5
      filename=['Subject',int2str(i),'\a',int2str(j),'t',int2str(k),'.mat'];
      ac=load(filename);
      N  = length(ac.sensor_readings);                                                     %采样点数
      n  = 0:N-1;
      t   = 0:1/Fs:1-1/Fs; 
      %figure(1);
      Wc=2*10/Fs;                                                                          %截止频率 10Hz
      %滤波后结果
      [b,a]=butter(5,Wc,'low');
      temp=filter(b,a,ac.sensor_readings);
      %均值
      fac1=mean(temp(:,1));
      fac2=mean(temp(:,4));
      %标准差
      fac3=std(temp(:,1),1,1);
      fac4to6=std(temp(:,4:6),1,1);
      %方差
      fac7=fac3.^2;
      fac8to10=fac4to6.^2;
      
      %中值
      fac11=median(temp(:,1));
      fac12=median(temp(:,4));
      
      %AI：T时间内三轴加速度的矢量和均值
      mi=sqrt(temp(:,1).^2+temp(:,2).^2+temp(:,3).^2);
      fac13=mean(mi);  
   
      %VI：T时间内mi的方差
      fac14=mean((mi-fac13).^2);
      
      %SMA：加速度信号的幅值
      sma=sum(abs(temp(:,1)))+sum(abs(temp(:,2)))+sum(abs(temp(:,3)));
      fac15=sma/N;
        
      %峰度
      fac16=kurtosis(temp(:,1));
      fac17=kurtosis(temp(:,2));
      fac18=kurtosis(temp(:,4));
      fac19=kurtosis(temp(:,5));
        
      %偏度
      fac20=skewness(temp(:,1));
        
      %四分位差
      fac21to26=iqr(temp);
      %周期加窗后的
      %r=[range(temp(21:220,:));range(temp(121:320,:));range(temp(221:420,:));range(temp(321:520,:))];
      %fac27to32=mean(r);  
      %25百分位点
      fac27to32=prctile(temp,25);
      %方差
      cor=corrcoef(temp);
      fac33=cor(2,1);
      fac34=cor(3,1);
      fac35=cor(3,2);
      fac36=cor(4,1);
      fac37=cor(4,2);
      fac38=cor(4,3);
      fac39=cor(5,1);
      fac40=cor(5,2);
      fac41=cor(5,3);
      fac42=cor(5,4);
      fac43=cor(6,1);
      fac44=cor(6,2);
      fac45=cor(6,3);
      fac46=cor(6,4);
      fac47=cor(6,5);
      
      
      %75百分位点
      fac48to53=prctile(temp,75);
      
      %根据乘坐电梯的竖直加速度变化，增加前半均值和后半均值
      fac54=10*mean(temp(1:N/2,1));
      fac55=10*mean(temp(N/2+1:N,1));
      fac56=max(temp(1:N/2,1));
      fac57=min(temp(N/2+1:N,1));
      
      %ADD
      %fac62=std(temp(:,2),1,1)^2;
      
      factor=[fac1,fac2,fac3,fac4to6,fac7,fac8to10,fac11,fac12,fac13,fac14,fac15,fac16,fac17,fac18,fac19,fac20,fac21to26,fac27to32];
      factor=[factor,fac33,fac34,fac35,fac36,fac37,fac38,fac39,fac40,fac41,fac42,fac43,fac44,fac45,fac46,fac47,fac48to53,fac54,fac55,fac56,fac57];
      data1=[data1;factor];
    end
  end
end

%% 数据归一化处理
[input,minI,maxI] = premnmx( data1');

%% 划分训练数据和测试数据
input=input';
traindata=input(1:720,:);
traindata=[traindata;input(781:840,:)];
testdata=input(721:780,:);

train_class=[];
test_class=[];
for x=1:13
  for y=1:12
    for z=1:5
      train_class=[train_class;y];
    end
  end
end
%%构造输出矩阵
%训练集
s1 = length(train_class) ;
train_output = zeros( s1 , 12  ) ;

for i = 1 : s1
   train_output( i , train_class( i )  ) = 1 ;
end

%测试集
for y=1:12
  for z=1:5
    test_class=[test_class;y];
  end
end

s2 = length(test_class) ;
test_output = zeros( s2 , 12  ) ;

for i = 1 : s2 
   test_output( i , test_class( i )  ) = 1 ;
end

%% 网络参数的初始化
numInputs = 57;
numHidden = 90;
numOutputs = 12;
w1 = rands(numHidden,numInputs);
b1 = rands(numHidden,1);
w2 = rands(numOutputs,numHidden);
b2 = rands(numOutputs,1);

%学习率
lr=0.05;
%迭代次数
epoch=1500;
%% 训练阶段
for i=1:epoch
    for j=1:20
        %隐藏层输出(1*h)sigmoid
        for k=1:numHidden
            hi(k)=w1(k,:)*traindata(j,:)'+b1(k);
            ho(k)=1/(1+exp(-hi(k)));
        end
        %输出层输出(1*o) pureline
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
        
        %更新权重和偏置值
        w1=w1+lr*dw1';
        b1=b1+lr*db1';
        w2=w2+lr*dw2;
        b2=b2+lr*db2;
    end
end
%% 测试阶段
py=[];
for i=1:s2
    %隐含层输出
    for k=1:numHidden
        pi(k)=w1(k,:)*testdata(i,:)'+b1(k);
        po(k)=1/(1+exp(-pi(k)));
    end
    %预测结果(输出层输出)
    py=[py,w2*po'+b2];
end

%% 统计识别正确率
[s11 , s22] = size( py ) ;
hitNum = 0 ;
result=[];
for i = 1 : s22
    [m , Index] = max( py( : ,  i ) ) ;
    result=[result;i,test_class(i),Index];
    if( Index  == test_class(i)   ) 
        hitNum = hitNum + 1 ; 
    end
end
sprintf('识别率是 %3.3f%%',100 * hitNum / s2 )
t=toc;
sprintf('%fminutes%',t/60)
