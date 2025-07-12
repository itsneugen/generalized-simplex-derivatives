%Author: Gabriel Jarry-Bolduc
%Date Created: April, 2020.
%Last Update: May 12, 2025.
%
%This function returns the generalized centered simplex gradient.
%
%Input:fun: function handle from R^n to R
%      x0: point of interest x0. Column vector in R^n
%       T: set of directions in matrix form  n \times m to add to x0
%       h: parameter to shrink T. Positive number.
%
%Output: CSGradValue: gcsg value at x0
%        info: info.AbsError: Absolute error between the true gradient and
%                             the gcsg at x0          
%              info.RelError: Relative error
%              info.gradient: True gradient at x0
%
function [CSGradValue,info] = gcsg(fun,x0,T,h)
%% Set h=1 if it is not specified
if (nargin<3)
   error('gcsg must have at least three inputs');
elseif (nargin==3)
  h=1;
end
%% Compute the gcsg
Sgval1=gsg(fun,x0,T,h);
Sgval2=gsg(fun,x0,-T,h);
CSGradValue=0.5*(Sgval1+Sgval2);
%% Calculate Absolute Error and relative error
if nargout==2
    n=length(x0);
    x=sym('x',[n,1]);
    Fun=fun(x);
    Gradient=gradient(Fun,x);
    for iter=1:n
        CellArrayVar{1,iter}=x(iter);
        CellArray{1,iter}=x0(iter); %  Create a cell since gradfun does not accept a vector as an input
    end
    Gradient_x0=double(subs(Gradient, CellArrayVar, CellArray));
    info.gradient=Gradient_x0';
    info.AbsError=double((norm(CSGradValue-Gradient_x0)));
    Norm=norm(Gradient_x0);
    if Norm>1e-16
        info.RelError=double(norm(CSGradValue-Gradient_x0)/norm(Gradient_x0));
    end
end
end
