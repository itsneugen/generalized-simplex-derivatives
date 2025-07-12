% Sample test script for gsg()

% Function to test
fun = @(x) x(1)^2 + 3*x(2)^2;  % A simple function f(x, y) = x^2 + 3y^2

% Point to evaluate gradient at
x0 = [1; 2];  % You can change this to other points later

% Simplex directions (for example, use canonical basis + extra direction)
T = [1 0 -1; 0 1 -1];  % 3 directions in R^2 that form a positive basis

% Step size (how far from x0 we step in each direction)
h = 1e-10;

% Call the GSG function
[SGradValue, info] = gsg(fun, x0, T, h);

% Print results
disp('Approximate gradient (GSG):');
disp(SGradValue);

disp('True gradient:');
disp(info.gradient);

disp('Absolute error:');
disp(info.AbsError);

disp('Relative error:');
disp(info.RelError);
