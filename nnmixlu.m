function [y,thismask] = nnmixlu(x,varargin)
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

opts.percentNeg = 10;
opts.mask = struct();
opts.mask.reluIndices = [];
opts.mask.negluIndices = [];

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end)) ;
else
  opts = vl_argparse(opts, varargin) ;
end
% determine mask
propNeg = opts.percentNeg/100 ;
thismask = opts.mask ;
if backMode && isempty(thismask)
  warning('nnmixlu: when using in backward mode, the mask should be specified') ;
end
if isempty(thismask)
  thismask.reluIndices = randperm(numel(x), floor((1-propNeg)*numel(x))) ;
  thismask.negluIndices = setdiff(1:numel(x),thismask.reluIndices) ;
end

if isa(x,'gpuArray')
  y = gpuArray(size(x)) ;
else
  y = single(size(x)) ;
end

if ~backMode
  y(thismask.reluIndices) = max(x(thismask.reluIndices), single(0)) ;
  y(thismask.negluIndices) = min(x(thismask.negluIndices), single(0)) ;
  y = reshape(y,size(x));
else
  y(thismask.reluIndices) = dzdy(thismask.reluIndices) .* (x(thismask.reluIndices) > single(0)) ;
  y(thismask.negluIndices) = dzdy(thismask.negluIndices) .* (x(thismask.negluIndices) > single(0)) ;
  y = reshape(y,size(x));
end