%Author: Gabriel Jarry-Bolduc
%Date created: September 2,2020
%Last Update: May 12, 2025
%
%This function returns the generalized simplex gradient. It is an
%approximation of the gradient at x0.
%
%Input: fun: function handle from R^n to R
%       x0:  point of interest in R^n (column vector)
%       T:   matrix of directions in R^n to add to x0. Each column is a
%            direcrion in R^n.
%Output: SGradValue: simplex gradient value at x0
%        info: info.gradient: true gradient at x0
%              info.AbsError: absolute error
%              info.RelError: relative error

function [SGradValue,info] = gsg(fun,x0,T,h)
%% Set Delta=1 if it is not specified
if (nargin==3)
  h=1;
end
[n,m]=size(T); %number of columns
%% Create the matrix T_h
T_h=h*T;
%% Create the matrix delta_f (row i :f(x^0+t^i)-f(x^0))
for col=1:m
    delta_f(col,:)=fun(x0+T_h(:,col))-fun(x0);
end
%% Calculate SGradValue
SGradValue=transpose(pinv(T_h))*delta_f;
%% Calculate Absolute Error
if nargout==2
    x=sym('x',[n,1]);
    Fun=fun(x);
    Gradient=gradient(Fun,x);
    for iter=1:n
        CellArrayVar{1,iter}=x(iter);
        CellArray{1,iter}=x0(iter); %  Create a cell since gradfun does not accept a vector as an input
    end
    Gradient_x0=double(subs(Gradient, CellArrayVar, CellArray));
    info.gradient=Gradient_x0';
    info.AbsError=double((norm(SGradValue-Gradient_x0)));
    Norm=norm(Gradient_x0);
    if Norm>1e-16
        info.RelError=double(norm(SGradValue-Gradient_x0)/norm(Gradient_x0));
    end
end
end

