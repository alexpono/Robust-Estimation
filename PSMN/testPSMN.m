% test PSMN

function varOut = testPSMN(plotornot)

load('allResults_auFilDeLEau.mat')
varOut = size(allresults,2);

switch plotornot
    case 'yes'
        figure
        plot([1 2],[2 4],'ks')
end

end

