clear all
close all;

warning off

p = 1;


% % 
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_1_28";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_5_28";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_10_28";     p = p+1;
path{p} = "Conv1D_trans_1_multi_removeTOF_shift_1";     p = p+1;
path{p} = "Conv1D_trans_1_multi_removeTOF_shift_5";     p = p+1;
path{p} = "Conv1D_trans_1_multi_removeTOF_shift_10";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_1_hindShift";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_5_hindShift";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_10_hindShift";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_1_hindShift2";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_5_hindShift2";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_10_hindShift2";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_1_dataBalance";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_5_dataBalance";     p = p+1;
% path{p} = "Conv1D_trans_1_multi_removeTOF_shift_10_dataBalance";     p = p+1;

% 

SET = "Set";
SET = "Set_hyperOnSet5_";



Label = [];
Output = [];


for s=  1:5
    subplot(5,1,s)
    hold off
    Output_ = [];
    for p = 1:length(path)
        load(SET+num2str(s)+"/"+path(p)+"/results/gradcam/output.txt")
        plot(output,'*-')
        hold on
        Output_ = [Output_,output];

        load(SET+num2str(s)+"/"+path(p)+"/results/gradcam/label.txt")



    end
            plot(diff(label'),'m--')
        legend(path,'Interpreter','latex')
        %     title(path{p},'Interpreter','latex')
        title("Cross Validation Set "+num2str(s),'Interpreter','latex')
        xlabel("measurment")
        ylabel("depth [mm]")
    Output = [Output;Output_];

    Label = [Label;label];
end


pause(1)
figure(2)
Depth = diff(Label')';
Error = abs(Output-Depth);
Diff_div = (Output-Depth);

subplot(2,1,1)
boxplot(Error);
yline([0,0.5,1],':')
title("Absolute Difference")
ylabel("Error [mm]")
% xticklabels(path)

pause(1)
subplot(2,1,2)
boxplot(Diff_div);
yline([-1:0.5:1],':')
ylabel("Error [mm]")
title("Difference")
xticklabels(path)
subplot(2,1,1)
xticklabels(path)
%
%  violin(Y)
disp("Mean Error: "+num2str(mean(Error)));
disp("Median Error:  "+num2str(median(Error)));


figure(3)

Label_div = [];
Error_div = [];
dist = 0.25;
for d = dist:dist:ceil(max(Depth)*(1/dist))*dist

    o = ((Depth>(d-dist)).*(Depth<d))>0;
    Label_div = [Label_div;d*ones(sum(o),1)];
    Error_div = [Error_div;Diff_div(o,:)];
end

for p = 1:length(path)
    subplot(length(path),1,p)
    boxplot(Error_div(:,p),Label_div)
    yline(0)
    yline([-2:1:2],':')
    yline([-1.5:1:1.5],':g')
    title(path{p},'Interpreter','latex')
    xlabel("depth [mm]")
    ylabel("error [mm]")
    axis([0,35,-2.5,2.5])
end


save("results","path","Output","Label")


