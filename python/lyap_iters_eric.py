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
		P1 = Q1
		P2 = Q2
		inv = np.linalg.inv
		norm = np.linalg.norm
		K1 = inv(R11 + B1.T @ P1 @ B1) @ (B1.T @ P1 @ A)
		K2 = inv(R22 + B2.T @ P2 @ B2) @ (B2.T @ P2 @ A)
		norms = []

		for _ in range(N):
			_K1 = K1
			_K2 = K2

			K1 = inv(R11 + B1.T @ P1 @ B1) @ (B1.T @ P1 @ (A - B2 @ _K2))
			K2 = inv(R22 + B2.T @ P2 @ B2) @ (B2.T @ P2 @ (A - B1 @ _K1))

			if True:
				P1 = (A - B1 @ K1 - B2 @ K2).T @ P1 @ (A - B1 @ K1 - B2 @ K2) + \
					 K1.T @ R11 @ K1 + K2.T @ R12 @ K2 + Q1
				P2 = (A - B1 @ K1 - B2 @ K2).T @ P2 @ (A - B1 @ K1 - B2 @ K2) + \
					 K1.T @ R21 @ K1 + K2.T @ R22 @ K2 + Q2

			else:
				P1 = (A - B2 @ K2).T @ P1 @ (A - B2 @ K2) - ((A - B2 @ K2).T @ P1 @ B1) @ \
					 inv(R11 + B1.T @ P1 @ B1) @ (B1.T @ P1 @ (A-B2 @ K2)) + Q1
				P2 = (A - B1 @ K1).T @ P2 @ (A - B1 @ K1) - ((A - B1 @ K1).T @ P2 @ B2) @ \
					 inv(R22 + B2.T @ P2 @ B2) @ (B2.T @ P2 @ (A-B1 @ K1)) + Q2


			norms.append([norm(P1), norm(P2)])
	else:
		test=False
		K1=0
		K2=0
		norms=0
	return [K1,K2], norms ,test
