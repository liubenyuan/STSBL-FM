function Result = STSBL_FM(PHI,Y,blkStartLoc,LearnLambda,varargin)
%------------------------------------------------------------------
% The code for FMLM optimized Spatio-Temporal Sparse Recovery
%
% Coded by: Liu Benyuan < liubenyuan AT gmail DOT com >
% Date    : 2013-12-12
%
% optimise : 
%        1. replace trace() with sum(diag())
%
%------------------------------------------------------------------
% Input for STSBL-FM:
%   PHI: projection matrix
%   Y:   CS measurements
%   blkStartLoc : Start location of each block
%   LearnLambda : (1) If LearnLambda = 1,
%                     use the lambda learning rule for MEDIUM SNR cases (SNR<20dB)
%                     (using lambda=std(y)*1e-1 or user-input value as initial value)
%                 (2) If LearnLambda = 2,
%                     use the lambda learning rule for HIGH SNR cases (SNR>=20dB)
%                     (using lambda=std(y)*1e-2 or user-input value as initial value)
%                 (3) If LearnLambda = 0, do not use the lambda learning rule
%                     ((using lambda=1e-6 or user-input value as initial value)
%
% [varargin values -- in most cases you can use the default values]
%   'LEARNTYPE'  : LEARNTYPE = 0: Ignore intra-block correlation
%                  LEARNTYPE = 1: Exploit intra-block correlation 
%                 [ Default: LEARNTYPE = 1 ]
%   'VERBOSE'    : debuging information.
%   'EPSILON'    : convergence criterion
%   'rb'         : temporal correlation
%
% ==============================  OUTPUTS ============================== 
%   Result :
%      Result.x          : the estimated block sparse signal
%      Result.gamma_used : indexes of nonzero groups in the sparse signal
%      Result.gamma_est  : the gamma values of all the groups of the signal
%      Result.B          : the final mean value of each correlation block
%      Result.count      : iteration times
%      Result.lambda     : the final value of lambda

% default values for STSBL-FM
eta = 1e-4;      % default convergence test
verbose = 0;     % print some debug information
learnType = 0;   % default not to exploit intra block correlation
max_it = 1000;   % maximum iterations
rb = 0.90;       % temporal smooth

% infer dimensions
[~,N] = size(PHI);
[~,T] = size(Y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. intialize, scale
scl = ones(1,T);
for i = 1 : T
    y = Y(:,i);
    scl(i) = 1.6 * std(y);
    Y(:,i) = y ./ scl(i);
end
SCL = diag(scl);

% select sigma2
stdy2 = mean(std(Y))^2;    % (norm(Y, 'fro')/T)^2
sigma2 = 1e-3*stdy2;       % default value if otherwise specified [99]
if LearnLambda == 0
    sigma2 = 1e-6;         % noiseless                            [0 ]
elseif LearnLambda == 2
    sigma2 = 1e-2*stdy2;   % high SNR (SNR>=20)                   [2 ]
elseif LearnLambda == 1
    sigma2 = 1e-1*stdy2;   % medium SNR (SNR<20)                  [1 ]
end

% process parameters
if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'learntype'
                learnType = varargin{i+1};
            case 'rb'
                rb = varargin{i+1};
            case 'epsilon'
                eta = varargin{i+1};
            case 'sigma2_scale'
                sigma2 = varargin{i+1}*stdy2;
            case 'max_iters'
                max_it = varargin{i+1};
            case 'verbose'
                verbose = varargin{i+1};
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end
    end
end

% calculate regularizer
beta = 1/sigma2;
ML = zeros(max_it,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. formalize the blocks and quantities used in the code
%    p           : the number of blocks
%    blkStartLoc : the start index of blk (i)
%    blkLenList  : the length of each block (i)
blkStartLoc = blkStartLoc(:);
p = length(blkStartLoc);
blkLenList = vertcat(blkStartLoc(2:end),N+1) - blkStartLoc;
maxLen = max(blkLenList);
minLen = min(blkLenList);
if maxLen == minLen,
    equalSize = 1;
else
    equalSize = 0;
end
% when the blkLen=1, or T=1, we avoid the exploiting feature.
if maxLen == 1, % spatio 1 blklen
    learnType = 0;
end
if T == 1, % temporal 1 blklen
    invB = 1;
else
    invB = eye(T)/genB(rb,T);
%     invB = eye(T)/temporalSmooth(rb,T);
end
%%% B = Y'*Y.*beta/M; % init
%%% DA = A{idx} - Am{idx};
%%% DB = q{idx}'/(eye(blkLenList(idx)) + DA*s{idx})*DA*q{idx};
%%% B = B - DB/(M*N); % iterative

%------------------------------------------------------------------------------
% now exact BSBL-FM except for the A_i update rules,                   by. liu
%------------------------------------------------------------------------------

% pre-allocating space
S          = cell(p,1); s = cell(p,1);
Q          = cell(p,1); q = cell(p,1);
currentSeg = cell(p,1);
localSeg   = cell(p,1);
Phi        = cell(p,1);
% 2. prepare the quantities used in the code.
for k = 1 : p
    currentLoc    = blkStartLoc(k);
    currentLen    = blkLenList(k);
    currentSeg{k} = currentLoc : 1 : (currentLoc + currentLen - 1);

    Phi{k} = PHI(:,currentSeg{k});
    S{k}   = beta.*Phi{k}'*Phi{k};
    Q{k}   = beta.*Phi{k}'*Y;
end

% 3. start from *NULL*, decide which one to add ->
A     = cell(p,1);
Am    = cell(p,1); % old A
theta = zeros(1,p);
for k = 1 : p
    Am{k}    = zeros(blkLenList(k));
    A{k}     = (S{k})\(Q{k}*Q{k}'/T - S{k})/(S{k});
    theta(k) = mean(real(diag(A{k}))); % note the real
    A{k}     = eye(blkLenList(k)).*theta(k);        % SIM reconstruct first
end

% select the basis that minimize the change of *likelihood*
ml  = inf*ones(1,p);
ig0 = find(theta>0);
for k = ig0
    ml(k) = Li(A{k},S{k},Q{k},blkLenList(k));
end
[~,index] = min(ml);
Am{index} = A{index}; % Am -> record the past value of A
if verbose, fprintf(1,'ADD,\t idx=%3d, GAMMA_OP=%f\n',index,theta(index)); end

% 3. update quantities (Sig,Mu,S,Q,Phiu)
Sigma_ii = (eye(blkLenList(index))/Am{index} + S{index})\eye(blkLenList(index));
Sig      = Sigma_ii;
Mu       = Sigma_ii*Q{index};
% The relevent block basis
Phiu = Phi{index};
for k = 1 : p
    Phi_k = Phi{k};
    PPKP  = Phi_k'*Phiu;
    S{k}  = S{k} - beta^2.*PPKP*Sigma_ii*PPKP';
    Q{k}  = Q{k} - beta  .*PPKP*Mu;
end

% now Loop
count = 0;
while (count<max_it)
    count = count + 1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    localLoc = 1;
    for i = 1 : length(index);
        k = index(i);
        localLen = blkLenList(k);
        localSeg{i} = localLoc:1:localLoc + localLen - 1;
        localLoc = localLoc + localLen;
    end

    % calculate s,q
    iog = setdiff(1:p,index);
    for k = index % the k-th basis is included
        invDenom = (eye(blkLenList(k)) - S{k}*Am{k})\eye(blkLenList(k));
        s{k} = invDenom*S{k};
        q{k} = invDenom*Q{k};
    end
    for k = iog % the k-th basis is not included
        s{k} = S{k};
        q{k} = Q{k};
    end

    % calculate A_i
    for k = 1 : p
        A{k} = (s{k})\(q{k}*invB*q{k}'/T - s{k})/(s{k});
        theta(k) = mean(real(diag(A{k})));
    end

    % regularize A_i
    if learnType == 2
        r_hat = averagedR(Sig,Mu,localSeg(1:length(index)));
        B_hat = genB(r_hat,maxLen);
    end
    for k = 1 : p
        if learnType == 0      % [0] without intra-correlation
            A{k} = eye(blkLenList(k))*theta(k);
        elseif learnType == 1  % [1] with individual intra corr
            rr = estimateR(A{k});
            Bc = genB(rr,blkLenList(k));
            A{k} = Bc*theta(k);
        elseif learnType == 2  % [2] with unified intra corr
            if equalSize == 1
                Bc = B_hat;
            else
                Bc = genB(r_hat,blkLenList(k));
            end
            A{k} = Bc.*theta(k);
        end
    end

    % choice the next basis that [minimizes] the cost function
    ml = inf*ones(1,p);
    for k = index
        if theta(k) > 0 % already IN, need [Re-estimate]
            ml(k) = Li(A{k},s{k},q{k},blkLenList(k)) ...
			        - Li(Am{k},s{k},q{k},blkLenList(k));
        else % already IN, need [Delete]
            ml(k) = -Li(Am{k},s{k},q{k},blkLenList(k));
        end
    end
    for k = iog
        if theta(k) > 0 % not IN, need [Add]
            ml(k) = Li(A{k},s{k},q{k},blkLenList(k));
        end
    end

    % as we are minimizing the cost function :
    [ML(count),idx] = min(ml);

    % check if terminates?
    if ML(count)>=0, break; end
    if count >= 2 && abs(ML(count)-ML(count-1)) < abs(ML(count)-ML(1))*eta, break; end

    % update blocks
    which = find(index==idx);
    % processing the quantities update
    if ~isempty(which)  % the select basis is already in the *LIST*
        seg    = localSeg{which};
        Sig_j  = Sig(:,seg);
        Sig_jj = Sig(seg,seg);
        if theta(idx)>0
            %--- re-estimate -------------------------------------------------------
            if verbose,fprintf(1,'REE,\t idx=%3d, GAMMA_OP=%f\n',idx,theta(idx));end
            ki  = Sig_j/(Sig_jj + Am{idx}/(Am{idx} - A{idx})*A{idx})*Sig_j';
            Sig = Sig - ki;
            Mu  = Mu - beta.*ki*Phiu'*Y;
            PKP = Phiu*ki*Phiu';
            for k = 1 : p
                Phi_m = Phi{k};
                PPKP  = Phi_m'*PKP;
                S{k}  = S{k} + beta^2.*PPKP*Phi_m;
                Q{k}  = Q{k} + beta^2.*PPKP*Y;
            end
            Am{idx} = A{idx};
        else
            %--- delete --------------------------------------------------------------
            if verbose,fprintf(1,'DEL,\t idx=%3d, GAMMA_OP=%f\n',idx,theta(which));end
            if length(index)==1, break; end % we are deleting the last one
            ki  = Sig_j/Sig_jj*Sig_j';
            Sig = Sig - ki;
            Mu  = Mu - beta.*ki*Phiu'*Y;
            PKP = Phiu*ki*Phiu';
            for k = 1 : p
                Phi_m = Phi{k};
                PPKP  = Phi_m'*PKP;
                S{k}  = S{k} + beta^2.*PPKP*Phi_m;
                Q{k}  = Q{k} + beta^2.*PPKP*Y;
            end
            % delete relevant basis and block
            index(which) = [];
            Mu(seg,:)    = [];
            Sig(:,seg)   = [];
            Sig(seg,:)   = [];
            Phiu(:,seg)  = [];
            %
            Am{idx}      = zeros(blkLenList(idx));
        end
    else
        if theta(idx)>0
            %--- add ---------------------------------------------------------------
            if verbose,fprintf(1,'ADD,\t idx=%3d, GAMMA_OP=%f\n',idx,theta(idx));end
            Phi_j     = Phi{idx};
            %
            Sigma_ii = (eye(blkLenList(idx))+A{idx}*S{idx})\A{idx};
            mu_i     = Sigma_ii*Q{idx};
            SPP      = Sig*Phiu'*Phi_j; % common
            Sigma_11 = Sig + beta^2.*SPP*Sigma_ii*SPP';
            Sigma_12 = -beta.*SPP*Sigma_ii;
            Sigma_21 = Sigma_12';
            mu_1     = Mu - beta.*SPP*mu_i;
            e_i      = Phi_j - beta.*Phiu*SPP;
            ESE      = e_i*Sigma_ii*e_i';
            for k = 1 : p
                Phi_m = Phi{k};
                S{k}  = S{k} - beta^2.*Phi_m'*ESE*Phi_m;
                Q{k}  = Q{k} - beta.*Phi_m'*e_i*mu_i;
            end
            % adding relevant basis
            Sig     = [Sigma_11 Sigma_12; ...
                       Sigma_21 Sigma_ii];
            Mu      = [mu_1; ...
                       mu_i];
            Phiu    = [Phiu Phi_j];
            index   = [index idx];
            Am{idx} = A{idx};
        else
            break; % null operation
        end
    end

end

% format the output ===> X the signal
weights = zeros(N,T);
formatSeg = [currentSeg{index}];
weights(formatSeg,:) = Mu;
Result.x = weights * SCL;
Result.r = 1.0; % lazy ...
Result.gamma_used = index;
Result.count = count;
Result.lambda = sigma2;
% END %

end

%------------------------- sub-functions --------------------------------------

% log-likelihood function
function val = Li(A,s,q,d) % you can optimize this function [time hunger]
    val = log(abs(det(eye(d) + A*s))) - sum(diag(real(q'/(eye(d) + A*s)*A*q)));
end

% estimate average correlation from \Sigma, \mu, blkLenList
function r = averagedR(Sig,Mu,localSeg)
    len = length(localSeg);
    ri = zeros(len,1);
    for i = 1 : len;
        seg      = localSeg{i};
        Sigma_ii = Sig(seg,seg);
        Mu_i     = Mu(seg);
        A        = Sigma_ii + Mu_i*Mu_i';
        ri(i)    = estimateR(A);
    end
    r = mean(ri);
end

% empirically estimate correlation r
function r = estimateR(A)
    L = size(A,1); ratio = L/(L-1);
    r = sum(diag(A,1)) ./ sum(diag(A)) * ratio;
    if abs(r) >= 0.98
        r = 0.98*sign(r);
    end
end

% generate Toeplitz Matrix according to r,len
function B = genB(r,len)
    jup = 0:len-1;
    bs = r.^jup;
    B = toeplitz(bs);
end

% generate temporal Smooth matrix
function [B,Bc] = temporalSmooth(r,len)
    A1 = eye(len);
    A2 = (-r).*[zeros(1,len-1) 0; eye(len-1), zeros(len-1,1)];
    Bc = A1 + A2;
    B = Bc'*Bc;
end


