clear all
close all;

warning off

p = 1;


results_linearApproximation = load("results_linearApproximation.mat");

path{p} = "_"; p = p+1;
path{p} = "Conv1D_trans_1_multi_removeTOF_shift_1";     p = p+1;
path{p} = "Conv1D_trans_1_multi_removeTOF_shift_5";     p = p+1;
path{p} = "Conv1D_trans_1_multi_removeTOF_shift_10";     p = p+1;
path{p} = "Conv1D_trans_1_multi_removeTOF_shift_5_28";     p = p+1;
path{p} = "Conv1D_trans_1_multi_removeTOF_shift_5_hindShift";     p = p+1;


NAME = {'CA','NN_1','NN_5','NN_{10}','NN_{5^s}','NN_{5^h}'};

%

SET = "Set";
SET = "Set_hyperOnSet5_";

lw = 2.5;
FontSize = 18;
lFontSize = 15;





figure(3)
tiledlayout(3,2);

for p = 1:length(path)
    nexttile
    Output_ = [];
    Label_ = [];
    if p == 1
        results_linearApproximation = load("results_linearApproximation.mat");
        Label_ = results_linearApproximation.Label;
        Output_ = results_linearApproximation.Output;
        Depth = Label_;
    else

        for s=  1:5



            load(SET+num2str(s)+"/"+path(p)+"/results/gradcam/output.txt")
            Output_ = [Output_;output];

            load(SET+num2str(s)+"/"+path(p)+"/results/gradcam/label.txt")
            Label_ = [Label_;label];


        end
        Label = Label_;
        Depth = diff(Label')';

    end

    Output = Output_ ;
    Diff_div = (Output-Depth); 

    




    Label_div = [];
    Error_div = [];
    dist = 0.25;

    QUANTILE_=[];
    for d = dist:dist:ceil(max(Depth)*(1/dist))*dist

        o = ((Depth>(d-dist)).*(Depth<d))>0;
        Label_div = [Label_div;d*ones(sum(o),1)];
        Error_div = [Error_div;Diff_div(o)];
  
            QUANTILE_ = [QUANTILE_;[median(Diff_div(o)),quantile(Diff_div(o),0.25),quantile(Diff_div(o),0.75)]];
   
    end

        
        disp(path{p})
        aError_div = abs(Error_div);
        disp("Error:")
        disp([median(aError_div),quantile(aError_div,0.25),quantile(aError_div,0.75),quantile(aError_div,0.75)-quantile(aError_div,0.25)])
        disp("Mean Distance")
        disp(mean([(QUANTILE_),(QUANTILE_(:,3)-QUANTILE_(:,2))]))
        disp("End Distance:")
        disp(([QUANTILE_(end,:),QUANTILE_(end,3)-QUANTILE_(end,2)]))

    disp("*********************")


    
    boxplot(Error_div,Label_div)
    set(gca,'FontSize',14)
    yline(0)
    yline([-2:1:2],':')
    yline([-1.5:1:1.5],':g')
    title(NAME{p}) %,'Interpreter','latex')
    xlabel("Depth [mm]",'FontSize',FontSize)
    ylabel("Difference [mm]",'FontSize',FontSize)
    axis([dist,2*dist+length(dist:dist:ceil(max(Depth)*(1/dist))*dist),min(Error_div(:)),max(Error_div(:))])
    if p == 1
        a = [dist,2*dist+length(dist:dist:ceil(max(Depth)*(1/dist))*dist),min(Error_div(:)),max(Error_div(:))];
    elseif p == 2
        axis(a)
    end






end
