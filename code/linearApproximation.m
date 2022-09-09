clear all, close all;

data = readtable("data_r.csv");
pixelSize = 10.86e-3;

Set_Train{1} = 0:1988; 
Set_Test{1} = 1989:2442;

Set_Train{2} = 554:2442; 
Set_Test{2} = 0:553;

Set_Train{3} = 1187:2996; 
Set_Test{3} = 554:1186;

Set_Train{4} = 1790:3629; 
Set_Test{4} = 1187:1789;

Set_Train{5} = 1989:4232; 
Set_Test{5} = 1790:1988;




DEPTH_new = [];
DEPTH_approximate = [];
for s =1:5
Modulo = length(data.Var2);

intervall_train = mod(Set_Train{s},Modulo)+1;
intervall_test = mod(Set_Test{s},Modulo)+1;


%% train 
depth = (data.Var3(intervall_train)-data.Var2(intervall_train))*pixelSize;
diff_depth = diff(depth); 
depth_step = median(diff_depth);


%% test
depth = (data.Var3(intervall_test)-data.Var2(intervall_test))*pixelSize;
depth_new = zeros(size(depth));
depth_approximate = zeros(size(depth));
for i = 1:length(depth)

    % first shot
%     data.Var1(i)
%     pause(0.1)
    if contains(data.Var1(intervall_test(i)),'001.lvm')
        starting_distance = depth(i);
        count = 0;
    end
    depth_new(i) = depth(i)-starting_distance;
    depth_approximate(i) = count*depth_step;
    count = count + 1;   


end

figure(1)
subplot(5,1,s)
plot(depth_new)
hold on
plot(depth_approximate)
legend("data","approximation")

DEPTH_new = [DEPTH_new;depth_new];
DEPTH_approximate = [DEPTH_approximate;depth_approximate];

end



figure(2)

Error = abs(DEPTH_approximate-DEPTH_new);
Diff_div = (DEPTH_approximate-DEPTH_new);

disp("Mean Error: "+num2str(mean(Error)));
disp("Median Error:  "+num2str(median(Error)));

subplot(2,1,1)
boxplot(Error');
yline([0,0.5,1],':')
title("Absolute Difference")
ylabel("Error [mm]")


pause(1)
subplot(2,1,2)
boxplot(Diff_div');
yline([-1:0.5:1],':')
ylabel("Error [mm]")
title("Difference")
subplot(2,1,1)
%
%  violin(Y)



Output = DEPTH_approximate;
Label = DEPTH_new;


save("results_linearApproximation","Output","Label")
