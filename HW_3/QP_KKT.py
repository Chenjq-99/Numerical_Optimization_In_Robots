import numpy as np;

def solve_kkt_system(Q, c, A, b, x_dim, constraint_dim):
    H_up = np.hstack((Q,A.T))
    H_low = np.hstack((A,np.zeros((constraint_dim, constraint_dim))))
    H = np.vstack((H_up, H_low))
    d = np.vstack((-c, b))
    res = np.linalg.solve(H,d)
    return res[:x_dim]


Q = np.array([[1,2,0],[0,2,4],[1,0,3]])

c = np.array([[1,2,3]]).T

A = np.array([[1,1,1],[2,1,3]])

b = np.array([[5,4]]).T

print(solve_kkt_system(Q,c,A,b,x_dim=3,constraint_dim=2))