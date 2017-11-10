from scipy.linalg import svd, eigh
from numpy import abs, dot, sum, empty
from numpy.random import randn
import matplotlib.pyplot as plt
from timeit import default_timer as timer

J = randn(768, 4096)
trials = 5  # 30

start = timer()
for n in range(trials):
    u, s, vh = svd(J, full_matrices=False, check_finite=False)
end = timer()
svd_time = end - start
print(' SVD elapsed time: %.2f s' % svd_time)

start = timer()
for n in range(trials):
    w, v = eigh(dot(J, J.T), overwrite_a=True, check_finite=False)
end = timer()
eigh_time = end - start
print('EIGH elapsed time: %.2f s' % eigh_time)

resid = empty(J.shape[0], dtype=float)
for n in range(J.shape[0]):
    resid[n] = abs(sum(v[:,-1-n]*u[:,n]))-1

print('SVD/EIGH: %.2f' % (svd_time/eigh_time))

fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(resid)
ax[1].plot(s**2, '+')
ax[1].plot(w[::-1], 'x')
plt.show()
