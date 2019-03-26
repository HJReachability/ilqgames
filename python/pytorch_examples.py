import torch
import numpy as np

x = torch.ones(2, 1, requires_grad=True)

# Try out some gradient descent.
print("Trying out some gradient descent.")
ii = 1
while True:
    y = torch.sum(x ** 2 - 1.5 * x[0, 0] + 2.4 * x[1, 0])
    y.backward()
    if x.grad.norm() < 1e-3 or ii > 1000:
        break

    a = 0.1 / ii**1.5
    ii += 1
    x.data -= a * x.grad.data
    if ii % 100 == 0:
        print(x)

# Try computing a full Jacobian.
print("Computing a Jacobian.")
def foo(x):
    out = torch.empty(2, 1)
    out[0, 0] = 5.0 * torch.sin(x[0, 0]) + 3.0 * x[1, 0] * x[1, 0]
    out[1, 0] = 5.0 * torch.cos(x[1, 0])
    return out

x = torch.ones(2, 1, requires_grad=True)
f = foo(x)
J = []
for ii in range(len(x)):
    J.append(torch.autograd.grad(f[ii], x, retain_graph=True)[0])

J = torch.cat(J, dim=1).detach().numpy().copy().T
print(J)

# Try computing a Hessian.
print("Trying to compute a Hessian.")
x = torch.ones(2, 1, requires_grad=True)
f = torch.sum(x ** 2)
print(x)
print(f)

f.backward(retain_graph=True,create_graph=True)
dx=x.grad
print(dx)

x.grad.data.zero_()
dx[0,0].backward(retain_graph=True)
hess0 = x.grad.detach().numpy().copy()

x.grad.data.zero_()
dx[1,0].backward(retain_graph=True)
hess1 = x.grad.detach().numpy().copy()

hess = np.concatenate([hess0, hess1], axis=1).T
print(hess)
