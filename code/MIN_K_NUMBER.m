function mink =MIN_K_NUMBER(array,k)
[~,index]=sort(array);
mink=index(1:k);