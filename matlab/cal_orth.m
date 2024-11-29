function re_orth = cal_orth(Z)
% input: Z is the produced embeding of SBL or other models
% output: ||Z'*Z||_F^2
Z = normalize(Z,'norm',2);
tempdim = size(Z,2);
temp1 = Z'*Z-eye(tempdim);
re_orth = sum(temp1 .* temp1,'all');
end

