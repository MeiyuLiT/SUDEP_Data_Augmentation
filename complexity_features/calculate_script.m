fileList = dir(fullfile('preprocessed_eeg', '*.mat'));%select simulate data or not
fileList = {fileList.name};

test = 'kol'; % select which feature you want to calculate
bob = 0;
result = struct;
for i=1:length(fileList)
    name = fileList{i};
    disp(name)
    load(strcat('preprocessed_eeg/', name))%select simulate data or not
    name = name(1:length(name)-4);
    fprintf(name);
    fprintf('\n')
    name = strrep(name, ' ','_');
    name = strrep(name, '.','_');
    name = strrep(name, '-','_');
    name = regexprep(name, '\(|\)', '');
    result.(name) = struct;
    fn = fieldnames(matData);
    fs = round(1/(matData.Time(2)-matData.Time(1)));
    % fs = 256;         %for simulate data
    for field=1:length(fn)
        fprintf(fn{field})
        fprintf(' ')
        if strcmp('Time', fn{field}) || strcmp('Sample', fn{field})
        % if strcmp('Time', fn{field}) || strcmp('x', fn{field}) %for simulate data
            continue
        end
        signal = matData.(fn{field});
        if sum(isnan(signal)) > 0
            bob = bob + 1;
        end
        signal(isnan(signal)) = 0;
        % signal = signal(1:round(length(signal)/2));
        signal = signal(round(length(signal)/2):end);
        if strcmp('dfa', test)
            res = DFA_fun(signal, 50:50:1000);
            result.(name).(fn{field}) = res(1);
        end

        if strcmp('hfd', test)
            res = hfd(signal, length(signal)/4);
            result.(name).(fn{field}) = res;
        end

        if strcmp('kol', test)
            signal = signal > mean(signal);
            kol = kolmogorov(signal);
            result.(name).(fn{field}) = kol;
        end

        if strcmp('ha', test)
            ha = jHjorthActivity(signal);
            result.(name).(fn{field}) = ha;
        end

        if strcmp('hm', test)
            ha = jHjorthMobility(signal);
            result.(name).(fn{field}) = ha;
        end

        if strcmp('hc', test)
            ha = jHjorthComplexity(signal);
            result.(name).(fn{field}) = ha;
        end

        if strcmp('gka', test)
            res = gka(signal);
            result.(name).(fn{field}) = res;
        end

        if strcmp('judd', test)
            res = judd(signal);
            result.(name).(fn{field}) = res;
        end

        if strcmp('band', test)
            delta = bandpower(signal, fs, [1 4]);
            theta = bandpower(signal, fs, [4 8]);
            alpha = bandpower(signal, fs, [8 12]);
            beta = bandpower(signal, fs, [12 30]);
            low_gamma = bandpower(signal, fs, [30 50]);
            high_gamma = bandpower(signal, fs, [50 99]);
            result.(name).(fn{field}) = [delta theta alpha beta low_gamma high_gamma]/bandpower(signal, fs, [1 99]);
        end

        if strcmp('theta', test)
            res = bandpower(signal, fs, [4 8]);
            result.(name).(fn{field}) = res;
        end

        if strcmp('alpha', test)
            res = bandpower(signal, fs, [8 12]);
            result.(name).(fn{field}) = res;
        end

        if strcmp('beta', test)
            res = bandpower(signal, fs, [12 30]);
            result.(name).(fn{field}) = res;
        end

        if strcmp('low_gamma', test)
            res = bandpower(signal, fs, [30 50]);
            result.(name).(fn{field}) = res;
        end

        if strcmp('high_gamma', test)
            res = bandpower(signal, fs, [50 99]);
            result.(name).(fn{field}) = res;
        end        
    end
end
fprintf('\n');
file_format = '%s_last_results.mat'; %change name for simulate
save(sprintf(file_format, test), 'result')

function HA = jHjorthActivity(X,~) 
sd = std(X); 
HA = sd ^ 2;
end


function[A,F] = DFA_fun(data,pts,order)
% -----------------------------------------------------
% DESCRIPTION:
% Function for the DFA analysis.
% INPUTS: 
% data: a one-dimensional data vector.
% pts: sizes of the windows/bins at which to evaluate the fluctuation
% order: (optional) order of the polynomial for the local trend correction.
% if not specified, order == 1;
% OUTPUTS: 
% A: a 2x1 vector. A(1) is the scaling coefficient "alpha",
% A(2) the intercept of the log-log regression, useful for plotting (see examples).
% F: A vector of size Nx1 containing the fluctuations corresponding to the
% windows specified in entries in pts.
% -----------------------------------------------------
% Checking the inputs
if nargin == 2
   order = 1; 
end
sz = size(data);
if sz(1)< sz(2)
    data = data';
end
exit = 0;
if min(pts) == order+1
        disp(['WARNING: The smallest window size is ' num2str(min(pts)) '. DFA order is ' num2str(order) '.'])
        disp('This severly affects the estimate of the scaling coefficient')
        disp('(If order == [] (so 1), the corresponding fluctuation is zero.)')
elseif min(pts) < (order+1)
        disp(['ERROR: The smallest window size is ' num2str(min(pts)) '. DFA order is ' num2str(order) ':'])
        disp(['Aborting. The smallest window size should be of ' num2str(order+1) ' points at least.'])
        exit = 1;
end
if exit == 1
    return
end
% DFA
npts = numel(pts);
F = zeros(npts,1);
N = length(data);
for h = 1:npts
    
    w = pts(h);
    
    n = floor(N/w);
    Nfloor = n*pts(h);
    D = data(1:Nfloor);
    
    y = cumsum(D-mean(D));
    
    bin = 0:w:(Nfloor-1);
    vec = 1:w;
    
    coeff = arrayfun(@(j) polyfit(vec',y(bin(j) + vec),order),1:n,'uni',0);
    y_hat = cell2mat(cellfun(@(y) polyval(y,vec),coeff,'uni',0));
    F(h)  = mean((y - y_hat').^2)^0.5;
    
end
A = polyfit(log(pts),log(F)',1);
end


% FUNCTION: kolmogorov.m
% DATE: 9th Feb 2005
% AUTHOR: Stephen Faul (stephenf@rennes.ucc.ie)
%
% Function for estimating the Kolmogorov Complexity as per:
% "Easily Calculable Measure for the Complexity of Spatiotemporal Patterns"
% by F Kaspar and HG Schuster, Physical Review A, vol 36, num 2 pg 842
%
% Input is a digital string, so conversion from signal to a digital stream
% must be carried out a priori
function complexity=kolmogorov(s)
n=length(s);
c=1;
l=1;
i=0;
k=1;
k_max=1;
stop=0;
while stop==0
	if s(i+k)~=s(l+k)
        if k>k_max
            k_max=k;
        end
        i=i+1;
        
        if i==l
            c=c+1;
            l=l+k_max;
            if l+1>n
                stop=1;
            else
                i=0;
                k=1;
                k_max=1;
            end
        else
            k=1;
        end
	else
        k=k+1;
        if l+k>n
            c=c+1;
            stop=1;
        end
	end
end
b=n/log2(n);
% a la Lempel and Ziv (IEEE trans inf theory it-22, 75 (1976), 
% h(n)=c(n)/b(n) where c(n) is the kolmogorov complexity
% and h(n) is a normalised measure of complexity.
complexity=c/b;
end