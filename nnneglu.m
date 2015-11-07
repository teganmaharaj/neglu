function y = nnneglu(x,dzdy)
% VL_NNNEGLU  CNN negative linear unit
%   Y = VL_NNNEGLU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNNEGLU(X, DZDY) computes the network derivative DZDX
%   with respect to the input X given the derivative DZDY with respect
%   to the output Y. DZDX has the same dimension as X.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNNEGLU(X), then
%   VL_NNNEGLU(X,DZDY) gives the same result as VL_NNNEGLU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.
%
% This function and neglu related parts Copyright (C) 2015 Tegan Maharaj
% Modified from or modeled on files from the VLFEAT library which is:
% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% 
% This file is sort of part of the VLFeat library, but it is original work 
% and they're not responsible for it. It is made available under
% the terms of the BSD license (see the COPYING file).

if nargin <= 1 || isempty(dzdy)
  y = min(x, single(0)) ;
else
  y = dzdy .* (x < single(0)) ;
end
