import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def integrate(ic, ti, p):
	m, k, req, xp, yp = p
	r1, v1, theta1, omega1, r2, v2, theta2, omega2 = ic

	sub = {M[0]:m[0], M[1]:m[1], R[0]:r1, Rdot[0]:v1, THETA[0]:theta1, THETAdot[0]:omega1, R[1]:r2, Rdot[1]:v2, THETA[1]:theta2, THETAdot[1]:omega2}
	for i in range(6):
		sub[X[i]] = xp[i]
		sub[Y[i]] = yp[i]
	for i in range(7):
		sub[K[i]] = k[i]
		sub[Req[i]] = req[i]

	diff_eq = [v1,A[0].subs(sub),omega1,ALPHA[0].subs(sub),v2,A[1].subs(sub),omega2,ALPHA[1].subs(sub)]

	print(ti)

	return diff_eq



t = sp.symbols('t')
M = sp.symbols('M0:2')
K = sp.symbols('K0:7')
Req = sp.symbols('Req0:7')
X = sp.symbols('X0:6')
Y = sp.symbols('Y0:6')
R = dynamicsymbols('R0:2')
THETA = dynamicsymbols('THETA0:2')

Rdot = [R[i].diff(t, 1) for i in range(2)]
THETAdot = [THETA[i].diff(t, 1) for i in range(2)]
Rddot = [R[i].diff(t, 2) for i in range(2)]
THETAddot = [THETA[i].diff(t, 2) for i in range(2)]

XR = [R[i] * sp.cos(THETA[i]) for i in range(2)]
YR = [R[i] * sp.sin(THETA[i]) for i in range(2)]
dR12 = sp.sqrt((XR[1] - XR[0])**2 + (YR[1] - YR[0])**2)

XRdot = [XR[i].diff(t, 1) for i in range(2)]
YRdot = [YR[i].diff(t, 1) for i in range(2)]

T = 0
for i in range(2):
	T += M[i] * (XRdot[i]**2 + YRdot[i]**2)
T *= sp.Rational(1, 2)
T = sp.simplify(T)

V = 0
for i in range(2):
	for j in range(3):
		dR = sp.sqrt((XR[i] - X[3 * i + j])**2 + (YR[i] - Y[3 * i + j])**2) - Req[3 * i + j]
		V += K[3 * i + j] * dR**2
V += K[6] * (dR12 - Req[6])**2
V *= sp.Rational(1, 2)
V = sp.simplify(V)

L = T - V

dL = []
ddot = []
for i in range(2):
	dLdR = L.diff(R[i], 1)
	dLdRdot = L.diff(Rdot[i], 1)
	ddtdLdRdot = dLdRdot.diff(t, 1)
	dL.append(ddtdLdRdot - dLdR)
	dLdTHETA = L.diff(THETA[i], 1)
	dLdTHETAdot = L.diff(THETAdot[i], 1)
	ddtdLdTHETAdot = dLdTHETAdot.diff(t, 1)
	dL.append(ddtdLdTHETAdot - dLdTHETA)
	ddot.append(Rddot[i])
	ddot.append(THETAddot[i])

sol = sp.solve(dL,ddot)

A = [sp.simplify(sol[i]) for i in Rddot]
ALPHA = [sp.simplify(sol[i]) for i in THETAddot]

#--------------------------------------------------------

m = np.array([1, 1])
k = np.array([25, 25, 25, 25, 25, 25, 25])
req = np.array([2.5, 2.5, 2.5 , 2.5, 2.5, 2.5, 2.5])
xp = np.array([-1.25, -3.75, -1.25, 1.25, 3.75, 1.25])
yp = np.array([0, 2.5, 5, 0, 2.5, 5])
r1o = 4
r2o = 2.5
v1o = 0
v2o = 0
theta1o = 125
theta2o = 30
omega1o = 0
omega2o = 0
mr = 0.25
tf = 30 

cnvrt = np.pi/180
theta1o *= cnvrt
theta2o *= cnvrt
omega1o *= cnvrt
omega2o *= cnvrt

p = m, k, req, xp, yp

ic = r1o, v1o, theta1o, omega1o, r2o, v2o, theta2o, omega2o

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args=(p,))

x = np.zeros((2,nframes))
y = np.zeros((2,nframes))
for i in range(2):
	x[i] = np.asarray([XR[i].subs({R[i]:rth[j,4 * i], THETA[i]:rth[j,4 * i + 2]}) for j in range(nframes)],dtype=float)
	y[i] = np.asarray([YR[i].subs({R[i]:rth[j,4 * i], THETA[i]:rth[j,4 * i + 2]}) for j in range(nframes)],dtype=float)


ke = np.asarray([T.subs({M[0]:m[0], M[1]:m[1], R[0]:rth[i,0], Rdot[0]:rth[i,1], THETA[0]:rth[i,2], THETAdot[0]:rth[i,3],\
		R[1]:rth[i,4], Rdot[1]:rth[i,5], THETA[1]:rth[i,6], THETAdot[1]:rth[i,7]}) for i in range(nframes)])
pe = np.asarray([V.subs({K[0]:k[0], K[1]:k[1], K[2]:k[2], K[3]:k[3], K[4]:k[4], K[5]:k[5], K[6]:k[6],\
		Req[0]:req[0], Req[1]:req[1], Req[2]:req[2], Req[3]:req[3], Req[4]:req[4], Req[5]:req[5], Req[6]:req[6],\
		 X[0]:xp[0], Y[0]:yp[0], X[1]:xp[1], Y[1]:yp[1], X[2]:xp[2], Y[2]:yp[2], X[3]:xp[3], Y[3]:yp[3], X[4]:xp[4], Y[4]:yp[4], X[5]:xp[5], Y[5]:yp[5],\
		 R[0]:rth[i,0], THETA[0]:rth[i,2], R[1]:rth[i,4], THETA[1]:rth[i,6]}) for i in range(nframes)])
E = ke + pe

#------------------------------------------------------------

xmax = x.max() + 2 * mr if x.max() > max(xp) else max(xp) + 2 * mr
xmin = x.min() - 2 * mr if x.min() < min(xp) else min(xp) - 2 * mr
ymax = y.max() + 2 * mr if y.max() > max(yp) else max(yp) + 2 * mr
ymin = y.min() - 2 * mr if y.min() < min(yp) else min(yp) - 2 * mr

r = np.zeros((7,nframes))
r[0:6] = np.asarray([np.sqrt((xp[i] - x[i//3])**2 + (yp[i] - y[i//3])**2) for i in range(6)])
r[6] = np.asarray([np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)])
rmax = np.asarray([max(r[i]) for i in range(7)])
theta = np.zeros((7,nframes))
theta[0:6] = np.asarray([np.arccos((yp[i] - y[i//3])/r[i]) for i in range(6)])
theta[6] = np.asarray([np.arccos((y[1] - y[0])/r[6])])
nl = np.asarray([int(np.ceil(i / (2 * mr))) for i in rmax])
l = np.zeros((7,nframes))
l[0:6] = np.asarray([(r[i] - mr)/nl[i] for i in range(6)])
l[6] = np.asarray([(r[6] - 2 * mr)/nl[6]])
h = np.sqrt(mr**2 - (0.5 * l)**2)
flipa = np.zeros((7,nframes))
flipb = np.zeros((7,nframes))
flipc = np.zeros((7,nframes))
for i in range(6):
	flipa[i] = np.asarray([-1 if x[i//3][j]>xp[i] and y[i//3][j]<yp[i] else 1 for j in range(nframes)])
	flipb[i] = np.asarray([-1 if x[i//3][j]<xp[i] and y[i//3][j]>yp[i] else 1 for j in range(nframes)])
	flipc[i] = np.asarray([-1 if x[i//3][j]<xp[i] else 1 for j in range(nframes)])
flipa[6] = np.asarray([-1 if x[0][i]>x[1][i] and y[0][i]<y[1][i] else 1 for i in range(nframes)])
flipb[6] = np.asarray([-1 if x[0][i]<x[1][i] and y[0][i]>y[1][i] else 1 for i in range(nframes)])
flipc[6] = np.asarray([-1 if x[0][i]<x[1][i] else 1 for i in range(nframes)])
xlo = np.zeros((7,nframes))
ylo = np.zeros((7,nframes))
for i in range(6):
	xlo[i] = x[i//3] + np.sign((yp[i] - y[i//3]) * flipa[i] * flipb[i]) * mr * np.sin(theta[i])
	ylo[i] = y[i//3] + mr * np.cos(theta[i])
xlo[6] = x[0] + np.sign((y[1] - y[0]) * flipa[6] * flipb[6]) * mr * np.sin(theta[6])
ylo[6] = y[0] + mr * np.cos(theta[6])
xl = np.zeros((7,max(nl),nframes))
yl = np.zeros((7,max(nl),nframes))
for i in range(6):
	xl[i][0] = xlo[i] + np.sign((yp[i]-y[i//3])*flipa[i]*flipb[i]) * 0.5 * l[i] * np.sin(theta[i]) - np.sign((yp[i]-y[i//3])*flipa[i]*flipb[i]) * flipc[i] * h[i] * np.sin(np.pi/2 - theta[i])
	yl[i][0] = ylo[i] + 0.5 * l[i] * np.cos(theta[i]) + flipc[i] * h[i] * np.cos(np.pi/2 - theta[i])
	for j in range(1,nl[i]):
		xl[i][j] = xlo[i] + np.sign((yp[i]-y[i//3])*flipa[i]*flipb[i]) * (0.5 + j) * l[i] * np.sin(theta[i]) - np.sign((yp[i]-y[i//3])*flipa[i]*flipb[i]) * flipc[i] * (-1)**j * h[i] * np.sin(np.pi/2 - theta[i])
		yl[i][j] = ylo[i] + (0.5 + j) * l[i] * np.cos(theta[i]) + flipc[i] * (-1)**j * h[i] * np.cos(np.pi/2 - theta[i])
xl[6][0] = xlo[6] + np.sign((y[1]-y[0])*flipa[6]*flipb[6]) * 0.5 * l[6] * np.sin(theta[6]) - np.sign((y[1]-y[0])*flipa[6]*flipb[6]) * flipc[6] * h[6] * np.sin(np.pi/2 - theta[6])
yl[6][0] = ylo[6] + 0.5 * l[6] * np.cos(theta[6]) + flipc[6] * h[6] * np.cos(np.pi/2 - theta[6])
for i in range(1,nl[6]):
	xl[6][i] = xlo[6] + np.sign((y[1]-y[0])*flipa[6]*flipb[6]) * (0.5 + i) * l[6] * np.sin(theta[6]) - np.sign((y[1]-y[0])*flipa[6]*flipb[6]) * flipc[6] * (-1)**i * h[6] * np.sin(np.pi/2 - theta[6])
	yl[6][i] = ylo[6] + (0.5 + i) * l[6] * np.cos(theta[6]) + flipc[6] * (-1)**i * h[6] * np.cos(np.pi/2 - theta[6])
xlf = x[1] - mr * np.sign((y[1]-y[0])*flipa[6]*flipb[6]) * np.sin(theta[6])
ylf = y[1] - mr * np.cos(theta[6])


fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	for i in range(2):
		circle=plt.Circle((x[i][frame],y[i][frame]),radius=mr,fc='xkcd:red')
		plt.gca().add_patch(circle)
	for i in range(6):
		circle=plt.Circle((xp[i],yp[i]),radius=0.5*mr,fc='xkcd:cerulean')
		plt.gca().add_patch(circle)
		plt.plot([xlo[i][frame],xl[i][0][frame]],[ylo[i][frame],yl[i][0][frame]],'xkcd:cerulean')
		plt.plot([xl[i][nl[i]-1][frame],xp[i]],[yl[i][nl[i]-1][frame],yp[i]],'xkcd:cerulean')
		for j in range(nl[i]-1):
			plt.plot([xl[i][j][frame],xl[i][j+1][frame]],[yl[i][j][frame],yl[i][j+1][frame]],'xkcd:cerulean')
	plt.plot([xlo[6][frame],xl[6][0][frame]],[ylo[6][frame],yl[6][0][frame]],'xkcd:cerulean')
	plt.plot([xl[6][nl[6]-1][frame],xlf[frame]],[yl[6][nl[6]-1][frame],ylf[frame]],'xkcd:cerulean')
	for i in range(nl[6]-1):
		plt.plot([xl[6][i][frame],xl[6][i+1][frame]],[yl[6][i][frame],yl[6][i+1][frame]],'xkcd:cerulean')
	plt.title("Springy Thingies: Springy Brings A Friend")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('springy_thingies.mp4', writer=writervideo)
plt.show()







