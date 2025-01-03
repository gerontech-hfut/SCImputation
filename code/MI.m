function mi = MI(a,b)
%����a��b�Ļ���Ϣ
% ����ȱʧֵ
missing_value = 9;

% ���˵�����ȱʧֵ�����ݵ�
valid_idx = ~(a == missing_value | b == missing_value);
a = a(valid_idx);
b = b(valid_idx);

%��ʼ������ء���ab��Ե�ؼ�������
hab = zeros(256,256);
ha = zeros(1,256);
hb = zeros(1,256);

%��һ��
if max(a)~=min(a)
    a = (a-min(a))/(max(a)-min(a));
else
    a = zeros(size(a));
end

if max(b)~=min(b)
    b = (b-min(b))/(max(b)-min(b));
else
    b = zeros(size(b));
end

a = double(int16(a*255))+1; 
b = double(int16(b*255))+1;

%ͳ��
for i=1:length(a) % ʹ�� length(a)
    indexx = a(i);
    indexy = b(i);
    hab(indexx,indexy) = hab(indexx,indexy)+1;%����
    ha(indexx) = ha(indexx)+1;%a
    hb(indexy) = hb(indexy)+1;%b
end

%����������Ϣ��
hsum = sum(sum(hab));
index = find(hab~=0);
p = hab/hsum;
Hab = sum(sum(-p(index).*log(p(index))));

%����a��Ϣ��
hsum = sum(sum(ha));
index = find(ha~=0);
p = ha/hsum;
Ha = sum(sum(-p(index).*log(p(index))));

%����b��Ϣ��
hsum = sum(sum(hb));
index = find(hb~=0);
p = hb/hsum;
Hb = sum(sum(-p(index).*log(p(index))));

%����a��b�Ļ���Ϣ
mi = Ha+Hb-Hab;

%����a��b�Ĺ�һ������Ϣ
%mi = hab/(Ha+Hb);