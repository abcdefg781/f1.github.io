import numpy as np

class RBF():
	def __init__(self,lbounds,ubounds,sigma,x_matrix,y_matrix):
		self.dimensions = np.zeros((len(lbounds),2))
		for i in range(len(lbounds)):
			self.dimensions[i,:] = [lbounds[i],ubounds[i]]
		self.lbounds = lbounds
		self.ubounds = ubounds
		self.sigma = sigma
		self.x_matrix = x_matrix
		self.y_matrix = y_matrix
		self.num_designs = x_matrix.shape[0]

		self.num_design_variables=len(lbounds)
		self.scaledDimensions = np.zeros((self.num_design_variables,2))
		self.scaledDimensions[:,1] = 1
		self.dimensionspans = []
		for i in range(self.dimensions.shape[0]):
			self.dimensionspans.append(self.dimensions[i][1]-self.dimensions[i][0])

		self.getMapping()
		
		self.getPhiMatrix()
		self.getWVector()


	def getPhiMatrix(self):
		phi_matrix = np.zeros((self.num_designs,self.num_designs))

		for i in range(self.num_designs):
			x_i = self.x_matrix[i][:]
			for j in range(self.num_designs):
				x_j = self.x_matrix[j][:]

				dist = np.linalg.norm(x_i-x_j,ord=2)
				phi_matrix[i][j] = self.basisfn(dist)
		self.phi_matrix = phi_matrix

	def basisfn(self,r):
		return np.exp(-r**2/(2*self.sigma**2))

	def getWVector(self):
		w = np.matmul(np.linalg.inv(self.phi_matrix),self.y_matrix)
		self.w = w

	def getPhiPredictionMatrix(self,x_guess):
		phi_prediction_matrix = np.zeros((self.num_designs))

		for i in range(self.num_designs):
			x_i = self.x_matrix[i][:]
			dist = np.linalg.norm(x_i-x_guess,ord=2)
			phi_prediction_matrix[i] = self.basisfn(dist)
		self.phi_prediction_matrix = phi_prediction_matrix

	def getGuess(self,x_guess):
		for i in range(len(x_guess)):
			x_guess[i] = (x_guess[i]-self.dimensions[i][0])/self.dimensionspans[i]
		
		self.getPhiPredictionMatrix(x_guess)
		y_guess = np.matmul(self.phi_prediction_matrix,self.w)
		return y_guess

	def getMapping(self): #change the x_matrix to be in the hypercube [0,1]^n
		new_x_matrix = np.copy(self.x_matrix)
		for i in range(self.x_matrix.shape[1]):
			new_x_matrix[:,i] = (self.x_matrix[:,i]-self.dimensions[i][0])/self.dimensionspans[i]
		self.x_matrix = new_x_matrix

	def getInverseMapping(self): #reverts the mapping of the x_matrix before sending data back to the user
		new_x_matrix = np.copy(self.x_matrix)
		for i in range(self.x_matrix.shape[1]):
			new_x_matrix[:,i] = (self.x_matrix[:,i]*self.dimensionspans[i])+self.dimensions[i][0]
		self.x_matrix = new_x_matrix

	