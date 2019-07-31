function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% ע�⣬�˴�z������������,���п����Ǿ������������
% ���ǴӾ���ĽǶȿ��ǿ��԰����������
% ���� 1+e^(-Z),exp(-z)��һ����������������Ҫ��1����Ϊ��exp(-z)ͬ���ľ���
% ����ʹ����ones(size(-z))
% Ȼ��S�ͺ�������Ҫ���������ھ�������Ҫ�Ծ����е�ÿһ��Ԫ�ؽ�������������ʹ�� ./

g = 1./(ones(size(-z))+exp(-z));
% =============================================================

end
