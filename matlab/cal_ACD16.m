function reacd = cal_ACD16(base,realbase)
% input: base is the produced embeding of SBL,
%        realbase is the eigenvectors calculated by eigs
% output: average cosine distance of 16 produced embedings
    tempdist3 = 1-pdist2(realbase',double(base'),'cosine');
    tempdist4 = 1-pdist2(realbase',-1 * double(base'),'cosine');
    tempdist5 = max(tempdist3,tempdist4);
    
    
    [aaa,tempmin] = max(tempdist5);
    [resa,resb] = sort(aaa,'descend');
    resc = tempmin(resb);
    tempdist5 = tempdist5(:,resb);
    
    tk = 1;
    while tk <= 16
        
        tempa = find(resc(1:tk-1) == resc(tk));
        if ~isempty(tempa)
            tempb = tempdist5(:,tk);
            [tempd,tempc] = sort(tempb,'descend');
            for tl = 2:size(realbase,2)
                if isempty(find(resc(1:tk-1) == tempc(tl)))
                    break;
                end
            end
            resc(tk) = tempc(tl);
            resa(tk) = tempd(tl);
            [resa,resd] = sort(resa,'descend');
            resc = resc(resd);
            tempdist5 = tempdist5(:,resd);
            if isempty(find(resc(1:tk-1) == resc(tk)))
                tk = tk+1;
            end
        else
            tk = tk+1;
        end
    end
    
    
    
    reacd = mean(resa(1:16));

end