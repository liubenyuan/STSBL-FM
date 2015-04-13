function Phi = genP(K,M,N)
% K : non-zero entries each row
% M : number of rows
% N : number of columns
%

% the size of the sensing matrix Phi
% CR = M/N;      % compression ratio
while 1
    Phi = zeros(M,N);
    for i = 1 : N
        ind = randperm(M);
        indice = ind(1:K);
        col = zeros(M,1);
        col(indice) = ones(K,1);
        Phi(:,i) = col;
    end
    
    if rank(Phi) == M
        break
    end
end