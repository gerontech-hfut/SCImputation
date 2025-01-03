function [RMSE]=RMSE_compute(A,B,F)%º∆À„À„∑®RMSE
basicnum=numel(find(A==9));
RMSE=sqrt(sum(sum(B-F).^2)/basicnum);