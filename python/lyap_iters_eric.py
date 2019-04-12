import numpy as np
import matplotlib.pyplot as plt
import copy

# Coupled Discrete Algebraic Riccati Equation solver. *** Does not work for non-zero R12 and R21 (I think).

def coupled_DARE_solve(A, B1, B2, Q1, Q2,
					   R11, R12, R21, R22, N=500):
	'''Solves the Coupled Algebraic Riccati Equation
	by Lyapunov iterations'''
	inv = np.linalg.inv
	S1 = B1 @ inv(R11) @ B1.T
	S2 = B2 @ inv(R22) @ B2.T


	n,m1=B1.shape
	n,m2=B2.shape
	test_mat1=np.zeros((n,n*m1))
	test_mat1[:,:m1]=B1[:,:m1]
	test_mat2=np.zeros((n,n*m2))
	test_mat2[:,0]=B2[:,0]

	mat=np.eye(n)
	for i in range(n):
		mat=mat.dot(A)
		test_mat1[:,(i+1)*m1:(i+2)*m1]=mat.dot(B1)[:,:m1]
		test_mat2[:,(i+1)*m2:(i+2)*m2]=mat.dot(B2)[:,:m2]

	# Check that the
	if np.linalg.matrix_rank(test_mat1)>=n or np.linalg.matrix_rank(test_mat2)>=n:
		test=True
		Z1 = Q1
		Z2 = Q2
		inv = np.linalg.inv
		norm = np.linalg.norm
		P1 = inv(R11 + B1.T @ Z1 @ B1) @ (B1.T @ Z1 @ A)
		P2 = inv(R22 + B2.T @ Z2 @ B2) @ (B2.T @ Z2 @ A)
		norms = []

		for _ in range(N):
			_P1 = P1
			_P2 = P2

			P1 = inv(R11 + B1.T @ Z1 @ B1) @ (B1.T @ Z1 @ (A - B2 @ _P2))
			P2 = inv(R22 + B2.T @ Z2 @ B2) @ (B2.T @ Z2 @ (A - B1 @ _P1))

			if True:
				Z1 = (A - B1 @ P1 - B2 @ P2).T @ Z1 @ (A - B1 @ P1 - B2 @ P2) + \
					 P1.T @ R11 @ P1 + P2.T @ R12 @ P2 + Q1
				Z2 = (A - B1 @ P1 - B2 @ P2).T @ Z2 @ (A - B1 @ P1 - B2 @ P2) + \
					 P1.T @ R21 @ P1 + P2.T @ R22 @ P2 + Q2

			else:
				Z1 = (A - B2 @ P2).T @ Z1 @ (A - B2 @ P2) - ((A - B2 @ P2).T @ Z1 @ B1) @ \
					 inv(R11 + B1.T @ Z1 @ B1) @ (B1.T @ Z1 @ (A-B2 @ P2)) + Q1
				Z2 = (A - B1 @ P1).T @ Z2 @ (A - B1 @ P1) - ((A - B1 @ P1).T @ Z2 @ B2) @ \
					 inv(R22 + B2.T @ Z2 @ B2) @ (B2.T @ Z2 @ (A-B1 @ P1)) + Q2


			norms.append([norm(Z1), norm(Z2)])
	else:
		test=False
		P1=0
		P2=0
		norms=0
	return [P1,P2], norms ,test
