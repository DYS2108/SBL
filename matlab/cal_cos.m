function re_cos = cal_cos(base,A)
% input: Z is the produced embeding of SBL or other models, A is the
%        normalized affinity matrix
% output: cosine between AZ and Z
temp1 = normalize(A * base,'norm',2);
temp2 = normalize(base,'norm',2);
temp3 = diag(temp1' * temp2);
temp3 = temp3 - 1;
re_cos = mean(temp3.*temp3);

end
