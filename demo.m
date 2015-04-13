% liu benyuan [liubenyuan@gmail.com] 
%
clear all;close all;

%==========================================================================
rng(1985,'v4');

% load TeraHertz data
load KAC;
myN = 128;
KAC = KangarooAndCoin(1:200,100:299);
f2=imresize(KAC,[myN myN]);

% generate sensing matrix (column wise)
N=myN; K=round(0.5*N);
% generate a Bernoulli sensing matrix with 2 non-zero entries each column
Phi = genP(2, K, N);
Phi = Phi./(ones(K,1)*sqrt(sum(Phi.^2)));

% recover in DFT basis
W = dftmtx(myN); W = W';
A = Phi*W;

% compress the data
y = Phi*f2;

%=============== for BSBL-FM ==============================================
blkStartLoc = [1:4:N];

tic;
    Result = STSBL_FM(A, y, blkStartLoc, 2, 'learnType', 0, 'epsilon', 1e-8, 'rb', 0.90);
runtime = toc;

%=== recover the coeff
fp = W*Result.x;

nmse = -20*log10(norm(fp-f2)/norm(f2));
fprintf('Runtime(s) = %f,\t NMSE(dB) = %f\n',runtime,nmse);

%% 
close all;

figure

ax1 = subplot(221);
imagesc(abs(f2)); colorbar; h1 = title('Amplitude-orginal');
set(ax1, 'LooseInset', get(ax1, 'TightInset'));

ax2 = subplot(222);
imagesc(abs(fp)); colorbar; h2 = title('Amplitude-recovered');
set(ax2, 'LooseInset', get(ax2, 'TightInset'));

ax3 = subplot(223);
imagesc(angle(f2)); colorbar; h3 = title('Phase-orginal');
set(ax3, 'LooseInset', get(ax3, 'TightInset'));

ax4 = subplot(224);
imagesc(angle(fp)); colorbar; h4 = title('Phase-recovered');
set(ax4, 'LooseInset', get(ax4, 'TightInset'));

set([ax1 ax2 ax3 ax4],'FontName','Times','FontSize',8);
set([ax1 ax2 ax3 ax4],...
    'Box','on','TickDir','out','TickLength',[.02 .02]); % 'XTick',xticks,
% set([hx1 hy1 hx2 hy2],'FontName','Times','FontSize',10,'FontWeight','bold');
set([h1 h2 h3 h4],'FontName','Times','FontSize',12,'FontWeight','bold');

% save fp_Thz_0_0.2_0.2.mat fp
