load('6.3&6.4.mat');
m = struct('string', {}, 'duration', {});
for c = 1:length(BOUT)
    d = BOUT{1,c}.duration;
    s = BOUT{1,c}.string;
    x.string = s;
    x.duration = d;
    m(c) = x;
end
mt = struct2table(m);
writetable(mt, '6.3&6.4.csv');
