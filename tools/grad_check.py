## This code is from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

q = torch.Tensor([[1,2,3]]) # 1x3
v = torch.Tensor([[[2,1,3],[3,2,1],[1,2,3]]]) #1x3x3

q = Variable(q, requires_grad=True)
v = Variable(v, requires_grad=True)

q_ = q.unsqueeze(1).repeat(1, 3, 1)
qv = (q_ * v).sum(2)
a = f.softmax(qv, 1)

v_ = (a.unsqueeze(2).expand(1,3,3) * v).sum(1)
output = (q * v_).sum(1)

output.backward()
print('autograd:')
print(q.grad.data)

# autograd
# q.grad = [1.0136  1.9155  3.0709]

s = torch.Tensor(3).fill_(0)
for i in range(3):
	dp = v.data[0][0].clone().fill_(0)
	p_i = a.data[0][i]
	for j in range(3):
		p_j = a.data[0][j]
		if i==j:
			dp = dp + p_i * (1 - p_i) * v.data[0][i]
		else:
			dp = dp - p_i * (p_j) * v.data[0][j]
	s = s + dp * (v.data[0][i] * q.data[0]).sum() 

dq = v_.data + s.unsqueeze(0)
print('for-loop:')
print(dq)

# for-loop
# dq = [1.0136  1.9155  3.0709]

p = a.data[0]
pp = torch.ger(p, p)
diag_p = torch.diag(p)

V = v.data[0] # V^T
T = torch.matmul((diag_p - pp), V)
qV = torch.matmul(V, q.data[0])
s = torch.matmul(qV, T)

dq = v_.data + s
print('linear algebra:') # (q^T V) (diag(p)-pp^T) V^T
print(dq)

T = torch.matmul(V.t(), diag_p - pp)
U = torch.matmul(T, V)
s = torch.matmul(U, q.data[0])

dq = v_.data + s
print('linear algebra:') # V (diag(p)-pp^T) V^T q
print(dq)

# linear algebra
# dq = [1.0136  1.9155  3.0709]
