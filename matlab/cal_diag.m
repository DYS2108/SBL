function re_diag = cal_diag(Z,A)
% input: Z is the produced embeding of SBL or other models, A is the
%        normalized affinity matrix
% output: off diag part of Z'AZ, smaller value indicates Z is more like the
%         eigenvactor
Z = normalize(Z,'norm',2);
temp1 = Z' * A * Z;
tempdim = size(Z,2);
temp1 = temp1 .* (1-eye(tempdim));
re_diag = sum(temp1 .* temp1,'all');
end

