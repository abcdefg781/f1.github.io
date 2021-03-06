import numpy as np
from scipy import interpolate
from scipy import signal
from Track import Track
import copy

def getTrackPoints(track,initial_direction=np.array([1,0])):
	pos = np.array([0,0])
	direction = initial_direction

	points = [[0,0]]

	for i in range(len(track.segments)):
		radius = track.getSegmentRadius(i)
		length = track.getSegmentLength(i)
		if radius==0:
			#on a straight
			endpoint = pos+direction*length

			for j in range(1,length.astype(int)-1):
				if j%5==0:
					points.append(pos+direction*j)

			pos = endpoint
		else:
			#corner
			#length is sweep in radians
			side = track.getCornerDirection(i)
			if side == 0:
				normal = np.array([-direction[1],direction[0]])
			else:
				normal = np.array([direction[1],-direction[0]])

			xc = pos[0]+radius*normal[0]
			yc = pos[1]+radius*normal[1]
			theta_line = np.arctan2(direction[1],direction[0])
			theta_0 = np.arctan2(pos[1]-yc,pos[0]-xc)
			if side == 0:
				theta_end = theta_0+length
				direction = np.array([np.cos(theta_line+length),np.sin(theta_line+length)])
			else:
				theta_end = theta_0-length
				direction = np.array([np.cos(theta_line-length),np.sin(theta_line-length)])
			theta_vector = np.linspace(theta_0, theta_end, 100)

			x,y = parametric_circle(theta_vector, xc, yc, radius)

			for j in range(len(x)):
				if j%10 == 0:
					points.append([x[j],y[j]])
			
			pos = np.array([x[-1],y[-1]])
			
	return np.array(points)

def parametric_circle(t,xc,yc,R):
    x = xc + R*np.cos(t)
    y = yc + R*np.sin(t)
    return x,y

def lerp(x,x0,x1,y0,y1):
	y = y0+(x-x0)*((y1-y0)/(x1-x0))
	return y

def getSpline(points,interval=0.00025,s=0.3):
	tck,u = interpolate.splprep(points.transpose(),s=s, k=5)
	unew = np.arange(0, 1.0, interval)
	finespline = interpolate.splev(unew, tck)

	gates = interpolate.splev(u, tck)
	gatesd = interpolate.splev(u, tck, der = 1)

	single = interpolate.splev(unew, tck,der=1)
	double = interpolate.splev(unew, tck,der=2)
	curv = (single[0]*double[1]-single[1]*double[0])/(single[0]**2+single[1]**2)**(3/2)

	return finespline,gates,gatesd,curv,single

def getSplineLength(spline):
	length = 0
	for i in range(1,len(spline[0])):
		prevpoint = [spline[0][i-1],spline[1][i-1]]
		currpoint = [spline[0][i],spline[1][i]]
		dx = np.sqrt((currpoint[0]-prevpoint[0])**2+(currpoint[1]-prevpoint[1])**2)
		length = length+dx
	return length


def getGateNormals(gates,gatesd):
	normals = []
	for i in range(len(gates[0])):
		der = [gatesd[0][i],gatesd[1][i]]
		mag = np.sqrt(der[0]**2+der[1]**2)
		normal1 = [-der[1]/mag,der[0]/mag]
		normal2 = [der[1]/mag,-der[0]/mag]

		normals.append([normal1,normal2])

	return normals

def getGateNormals2(single):
	single = np.array(single)
	mag = np.sqrt(single[0,:]**2+single[1,:]**2)
	single = single/mag

	normal1 = np.concatenate([[-single[1,:]],[single[0,:]]],axis=0)
	normal2 = np.concatenate([[single[1,:]],[-single[0,:]]],axis=0)

	return np.hstack((normal1.T,normal2.T))

def transformGates(gates):
	#transforms from [[x positions],[y positions]] to [[x0, y0],[x1, y1], etc..]
	newgates = np.array(gates).T
	return newgates

def reverseTransformGates(gates):
	#transforms from [[x0, y0],[x1, y1], etc..] to [[x positions],[y positions]]
	newgates = np.array(gates).T
	return newgates

def displaceSpline(gateDisplacements,finespline,normals):
	# displacedSpline = np.zeros(np.array(finespline).shape)
	# for i in range(len(finespline[0])):
	# 	normal = normals[i][0]
	# 	displacedSpline[:,i] = [finespline[0][i]+normal[0]*gateDisplacements[i],finespline[1][i]+normal[1]*gateDisplacements[i]]
	normalDisplacements = np.multiply(np.array(normals)[:,0],np.array(gateDisplacements)[:,None])
	displacedSpline = np.array(finespline)+normalDisplacements.T
	return displacedSpline

# def setGateDisplacements(gateDisplacements,gates,normals):
# 	#does not modify original gates, returns updated version
# 	newgates = np.copy(gates)
# 	for i in range(len(gates[0])):
# 		if i > len(gateDisplacements)-1:
# 			disp = 0
# 		else:
# 			disp = gateDisplacements[i]
# 		#if disp>0:
# 		normal = normals[i][0] #always points outwards
# 		#else:
# 		#	normal = normals[i][1] #always points inwards
# 		newgates[0][i] = newgates[0][i] + disp*normal[0]
# 		newgates[1][i] = newgates[1][i] + disp*normal[1]
# 	return newgates

def getArea(point1,point2,point3):
	area = np.abs(0.5*(point1[0]*point2[1]+point2[0]*point3[1]+point3[0]*point1[1]-point1[1]*point2[0]-point2[1]*point3[0]-point3[1]*point1[0]))
	return np.abs(area)

def getLength(point1,point2):
	return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

def getApex(curvature):
	apex = signal.find_peaks(curvature,height=0.005,distance=40)
	return apex



