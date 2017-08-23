%% --------------------------------------------------
% ����BP������������˶�״̬ʶ��
% ��.Ԥ����:������˹��ͨ�˲������봦��
% ��.������ȡ��ѡ��57ά����
% ��.����ʶ��BP����������������ز�+����㣩
%  ��Ԫ��������㣨57�����ز㣨90������㣨12��
%  ѧϰ�����ݶ��½���
%  ���亯�������ز㣨sigmoid�������������(���Ժ���)
%  ѵ������13�������ߡ�12��������4��ʱ�̹�780���ɼ�����
%  ���Լ�������60������
% ----------------------------------------------------------- 

clear;clc;
data1=[];
tic;
%% Ԥ����+������ȡ
Fs = 100;                                                                  %������
for i=1:14
  for j=1:12
    for k=1:5
      filename=['Subject',int2str(i),'\a',int2str(j),'t',int2str(k),'.mat'];
      ac=load(filename);
      N  = length(ac.sensor_readings);                                                     %��������
      n  = 0:N-1;
      t   = 0:1/Fs:1-1/Fs; 
      %figure(1);
      Wc=2*10/Fs;                                                                          %��ֹƵ�� 10Hz
      %�˲�����
      [b,a]=butter(5,Wc,'low');
      temp=filter(b,a,ac.sensor_readings);
      %��ֵ
      fac1=mean(temp(:,1));
      fac2=mean(temp(:,4));
      %��׼��
      fac3=std(temp(:,1),1,1);
      fac4to6=std(temp(:,4:6),1,1);
      %����
      fac7=fac3.^2;
      fac8to10=fac4to6.^2;
      
      %��ֵ
      fac11=median(temp(:,1));
      fac12=median(temp(:,4));
      
      %AI��Tʱ����������ٶȵ�ʸ���;�ֵ
      mi=sqrt(temp(:,1).^2+temp(:,2).^2+temp(:,3).^2);
      fac13=mean(mi);  
   
      %VI��Tʱ����mi�ķ���
      fac14=mean((mi-fac13).^2);
      
      %SMA�����ٶ��źŵķ�ֵ
      sma=sum(abs(temp(:,1)))+sum(abs(temp(:,2)))+sum(abs(temp(:,3)));
      fac15=sma/N;
        
      %���
      fac16=kurtosis(temp(:,1));
      fac17=kurtosis(temp(:,2));
      fac18=kurtosis(temp(:,4));
      fac19=kurtosis(temp(:,5));
        
      %ƫ��
      fac20=skewness(temp(:,1));
        
      %�ķ�λ��
      fac21to26=iqr(temp);
      %���ڼӴ����
      %r=[range(temp(21:220,:));range(temp(121:320,:));range(temp(221:420,:));range(temp(321:520,:))];
      %fac27to32=mean(r);  
      %25�ٷ�λ��
      fac27to32=prctile(temp,25);
      %����
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
      
      
      %75�ٷ�λ��
      fac48to53=prctile(temp,75);
      
      %���ݳ������ݵ���ֱ���ٶȱ仯������ǰ���ֵ�ͺ���ֵ
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

%% ���ݹ�һ������
[input,minI,maxI] = premnmx( data1');

%% ����ѵ�����ݺͲ�������
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
%%�����������
%ѵ����
s1 = length(train_class) ;
train_output = zeros( s1 , 12  ) ;

for i = 1 : s1
   train_output( i , train_class( i )  ) = 1 ;
end

%���Լ�
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

%% ��������ĳ�ʼ��
numInputs = 57;
numHidden = 90;
numOutputs = 12;
w1 = rands(numHidden,numInputs);
b1 = rands(numHidden,1);
w2 = rands(numOutputs,numHidden);
b2 = rands(numOutputs,1);

%ѧϰ��
lr=0.05;
%��������
epoch=1500;
%% ѵ���׶�
for i=1:epoch
    for j=1:20
        %���ز����(1*h)sigmoid
        for k=1:numHidden
            hi(k)=w1(k,:)*traindata(j,:)'+b1(k);
            ho(k)=1/(1+exp(-hi(k)));
        end
        %��������(1*o) pureline
        yi=w2*ho'+b2;
        yo=yi;
        %�����������Ԫ���ݶ���(1*o)
        gj=train_output(j,:)-yo';
        %��������������
        dw2=gj'*ho;
        db2=gj';
        %�������ز���Ԫ���ݶ���(1*h)
        for k=1:numHidden
            sum(k)=gj*w2(:,k);
        end
        eh=ho.*(1-ho);
        %���ز����������
        for m=1:1:numInputs
            for n=1:1:numHidden
                dw1(m,n)=eh(n)*traindata(j,m)*sum(n);
                db1(n)=eh(n)*sum(n);
            end
        end 
        
        %����Ȩ�غ�ƫ��ֵ
        w1=w1+lr*dw1';
        b1=b1+lr*db1';
        w2=w2+lr*dw2;
        b2=b2+lr*db2;
    end
end
%% ���Խ׶�
py=[];
for i=1:s2
    %���������
    for k=1:numHidden
        pi(k)=w1(k,:)*testdata(i,:)'+b1(k);
        po(k)=1/(1+exp(-pi(k)));
    end
    %Ԥ����(��������)
    py=[py,w2*po'+b2];
end

%% ͳ��ʶ����ȷ��
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
sprintf('ʶ������ %3.3f%%',100 * hitNum / s2 )
t=toc;
sprintf('%fminutes%',t/60)
