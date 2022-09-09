% clear all; close all;

lw = 2.5;
FontSize = 18;
lFontSize = 15;



results = load("results.mat");
results_linearApproximation = load("results_linearApproximation.mat");
NAME = {'CA','NN_1','NN_5','NN_{10}'};

Output = results.Output;
Label = results.Label(:,2)-results.Label(:,1);
Label = Label.*ones(size(Output));
% 
% Output = [Output(:,1),Output];
% Label  = [Label(:,1),Label];
Output = [results_linearApproximation.Output,Output];
Label  = [results_linearApproximation.Label,Label];


Error_div = Output-Label;

figure(1)
subplot(2,1,1)
boxplot(Error_div,'Labels',NAME)
yline([0,0.5,-0.5],":b")
ylabel("Error [mm]")


subplot(2,1,2)
boxplot(abs(Error_div),'Labels',NAME)
yline([0,0.5],":b")
ylabel("Absolute Error [mm]")



disp("Mean Absolute Error: "+num2str(mean(abs(Error_div))));
disp("Varianz Absolute Error:  "+num2str(var(abs(Error_div))));
disp("**********************************************************")
disp("Median Absolute Error:  "+num2str(median(abs(Error_div))));
disp("Quantile 0.25-0.75 Absolute Error:  "+num2str(quantile(abs(Error_div),0.75)-quantile(abs(Error_div),0.25)));
disp("Quantile 0.25 Absolute Error:  "+num2str(quantile(abs(Error_div),0.25)));
disp("Quantile 0.75 Absolute Error:  "+num2str(quantile(abs(Error_div),0.75)));
disp("**********************************************************")
% a = abs(Error_div);
% disp("Cohen's d absolute: "+num2str(mean(a)./sqrt(var(a))))
% a = (Error_div);
% disp("Cohen's d: "+num2str(mean(a)./sqrt(var(a))))


New_median = [];
New_quantile_25 = [];
New_quantile_75 = [];
NumberOfData = [];
figure(2)

dist = 0.25;
tiledlayout(2,2);
for p = 1:size(Label,2)
    error_ = [];
    label_ = [];
    QUANTILE_ = [];
    for d = dist:dist:ceil(max(Label(:,p))*(1/dist))*dist

        o = ((Label(:,p)>(d-dist)).*(Label(:,p)<d))>0;
        label_ = [label_;d*ones(sum(o),1)];
        error_ = [error_;Error_div(o,p)];
        QUANTILE_ = [QUANTILE_;[median(Error_div(o,p)),quantile(Error_div(o,p),0.25),quantile(Error_div(o,p),0.75)]];
    end
%     subplot(size(Label,2)/2,2,p)
    mean([(QUANTILE_),(QUANTILE_(:,3)-QUANTILE_(:,2))])
    ([QUANTILE_(end,:),QUANTILE_(end,3)-QUANTILE_(end,2)])
    nexttile
    boxplot(error_,label_)
    set(gca,'FontSize',14)
    yline(0)
    yline([-2:1:2],':')
    yline([-1.5:1:1.5],':g')
    title(NAME{p}) %,'Interpreter','latex')
    xlabel("Depth [mm]",'FontSize',FontSize)
    ylabel("Difference [mm]",'FontSize',FontSize)
    axis([0,1+length(0.1:dist:ceil(max(Label(:,p))*(1/dist))*dist),min(Error_div(:)),max(Error_div(:))])



    new_median = [];
    new_quantile_25 = [];
    new_quantile_75 = [];
    numberOfData = [];
    for i = unique(label_')

        o = label_==i;

        new_median = [new_median;median((error_(o)))];
        new_quantile_25 = [new_quantile_25;quantile((error_(o)),0.25)];
        new_quantile_75 = [new_quantile_75;quantile((error_(o)),0.75)];
        numberOfData = [numberOfData;sum(o)];

    end
    New_median = [New_median,new_median];
    New_quantile_25 = [New_quantile_25,new_quantile_25];
    New_quantile_75 = [New_quantile_75,new_quantile_75];
    NumberOfData = [NumberOfData,numberOfData];




end

disp("New Median Absolute Error: "+num2str(mean(abs(New_median))));
New_quantile = abs(New_quantile_25-New_quantile_75);
disp("Mean Quantile Length:  "+num2str(mean(New_quantile)));


figure(5)
subplot(2,2,1)
plot(unique(label_'),abs(New_median))
title("absolute median Value")
legend(NAME)
subplot(2,2,2)
plot(unique(label_'),New_quantile)
title("Quantile length")
legend(NAME)
subplot(2,2,3)
plot(unique(label_'),abs(New_median).*New_quantile)
legend(NAME)
title("Quantile*abs(median)")
subplot(2,2,4)
plot(unique(label_'),NumberOfData)
legend(NAME)
title("number of data points")


figure(3)

for p = 1:size(Label,2)
    error_ = [];
    label_ = [];
    for d = 0.1:dist:ceil(max(Label(:,p))*(1/dist))*dist

        o = ((Label(:,p)>(d-dist)).*(Label(:,p)<d))>0;
        label_ = [label_;d*ones(sum(o),1)];
        error_ = [error_;abs(Error_div(o,p))];
    end
    subplot(size(Label,2)/2,2,p)
    boxplot(error_,label_)
    yline(0)
    yline([-2:1:2],':')
    yline([0:0.25:0.9],':g')
    title(NAME{p}) %,'Interpreter','latex')
    xlabel("Depth [mm]")
    ylabel("Error [mm]")
    axis([0,1+length(0.1:dist:ceil(max(Label(:,p))*(1/dist))*dist),0,max(Error_div(:))])







end

figure(4)
I = [501,1000];
for p = 1:size(Label,2)
    subplot(size(Label,2)/2,2,p)
    hold off
    plot(Label(:,p),"--")
    hold on
    plot(Output(:,p))
    title(NAME{p}) %,'Interpreter','latex')
    ylabel("Depth [mm]")
    axis([I(1),I(2),0,4])
end






