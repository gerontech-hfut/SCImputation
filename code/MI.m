function mi = MI(a,b)
%计算a和b的互信息
% 定义缺失值
missing_value = 9;

% 过滤掉含有缺失值的数据点
valid_idx = ~(a == missing_value | b == missing_value);
a = a(valid_idx);
b = b(valid_idx);

%初始化混合熵、和ab边缘熵计算数组
hab = zeros(256,256);
ha = zeros(1,256);
hb = zeros(1,256);

%归一化
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

%统计
for i=1:length(a) % 使用 length(a)
    indexx = a(i);
    indexy = b(i);
    hab(indexx,indexy) = hab(indexx,indexy)+1;%联合
    ha(indexx) = ha(indexx)+1;%a
    hb(indexy) = hb(indexy)+1;%b
end

%计算联合信息熵
hsum = sum(sum(hab));
index = find(hab~=0);
p = hab/hsum;
Hab = sum(sum(-p(index).*log(p(index))));

%计算a信息熵
hsum = sum(sum(ha));
index = find(ha~=0);
p = ha/hsum;
Ha = sum(sum(-p(index).*log(p(index))));

%计算b信息熵
hsum = sum(sum(hb));
index = find(hb~=0);
p = hb/hsum;
Hb = sum(sum(-p(index).*log(p(index))));

%计算a和b的互信息
mi = Ha+Hb-Hab;

%计算a和b的归一化互信息
%mi = hab/(Ha+Hb);