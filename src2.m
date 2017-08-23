%% ---------------------------------------------------
% 
%       ����BP������������������˶�״̬��ʶ��
% 
% ----------------���������繤��ʵ��---------------- 

clear;clc;
data1=[];
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

%% ��һ��
[input,minI,maxI] = premnmx( data1')

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
%�����������
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

%����������
net = newff( minmax(traindata') , [100 12] , { 'logsig' 'purelin' } , 'traingdx' ) ; 

%����ѵ������
net.trainparam.show = 100 ;
net.trainparam.epochs = 1500 ;
net.trainparam.goal = 0.01 ;
net.trainParam.lr = 0.01 ;
net.divideFcn = '';

%��ʼѵ��
net = train( net, traindata' , train_output' ) ;

%����
Y = sim( net , testdata' ) ;

%ͳ��ʶ����ȷ��
[s1 , s2] = size( Y ) ;
hitNum = 0 ;
result=[];
for i = 1 : s2
    [m , Index] = max( Y( : ,  i ) ) ;
    result=[result;i,test_class(i),Index];
    if( Index  == test_class(i)   ) 
        hitNum = hitNum + 1 ; 
    end
end
sprintf('ʶ������ %3.3f%%',100 * hitNum / s2 )
%% ����δ֪����
data3=[];
unknow_factor=[];
for i=1:20
   filename=['unknow\unknow',int2str(i),'.mat'];
   unknow_ac=load(filename);
   N = length(unknow_ac.sensor_readings);                                                     %��������
   n = 0:N-1;
   t = 0:1/Fs:1-1/Fs; 
   %figure(1);
   Wc=2*10/Fs;                                                            %��ֹƵ�� 10Hz
   [b,a]=butter(5,Wc,'low');
   %�˲�����
   temp=filter(b,a,unknow_ac.sensor_readings);  
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
      
      %r=[range(temp(21:220,:));range(temp(121:320,:));range(temp(221:420,:));range(temp(321:520,:))];
      %fac27to32=mean(r);  
      
      %25�ٷ�λ��
      fac27to32=prctile(temp,25);
      
      %���ϵ��
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

      
      unknow_factor=[fac1,fac2,fac3,fac4to6,fac7,fac8to10,fac11,fac12,fac13,fac14,fac15,fac16,fac17,fac18,fac19,fac20,fac21to26,fac27to32];
      unknow_factor=[unknow_factor,fac33,fac34,fac35,fac36,fac37,fac38,fac39,fac40,fac41,fac42,fac43,fac44,fac45,fac46,fac47,fac48to53,fac54,fac55,fac56,fac57];
      data3=[data3;unknow_factor];
end
Y2 = sim( net , data3' ); 

%ʶ��unknow��
[s1 , s2] = size( Y2) ;
unknow_result=[];
for i = 1 : s2
    [m , Index] = max( Y2( : ,  i ) ) ;
    unknow_result=[unknow_result;i,Index];
    for j=1 : s1
        if j==Index
            Y2(j,i)=1;
        else
            Y2(j,i)=0;
        end
    end
end
Y2;