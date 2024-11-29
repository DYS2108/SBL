%%
%*************************calculate the ACD16 of produced embeding
clear;
files = dir('path of affinity matrix and laplacian matrix');
%
calbasenum = 16;
totalacc = zeros(182,calbasenum);

for ti = 3:184
    
    tempname = files(ti).name;
    %load(['path of features',tempname]);
    load(['path of affinity matrix and laplacian matrix',tempname]);
    t_res = load(['path of result',tempname]);
    
    base = double(squeeze(t_res.re_base));
    
    testvalue = base' * full(normlap) * base;
    
    testvalue = diag(testvalue);
    [val,temppp] = sort(testvalue);
    base = base(:,temppp);
    
    
    [realbase, ~] = eigs(normlap,200,'smallestabs');
    realbase = realbase(:,1:200);
    
    
    totalacc(ti,:) = cal_ACD16(base,realbase);
    
    
end

mean(totalacc(3:184,:));
std(totalacc(3:184,:));



