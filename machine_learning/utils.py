from numpy import matrix
from numpy import linalg
import math
import matplotlib.pyplot as plt
import random

class FunctionGenerator:

	@staticmethod
	def binary_classifier(weight_vector):
		def _binary_classifier(x_vector):
			num = 0.0	
			for i in range(len(x_vector)):
				num += x_vector[i] * _binary_classifier.weight_vector[i]
			if num > 0:
				return 1
			return -1
			
		_binary_classifier.weight_vector = weight_vector
		return _binary_classifier
		
	@staticmethod
	def nonlinear_binary_classifier(weight_vector):
		def _f(x_vector):
			num = 0.0	
			for i in range(len(x_vector)):
				num += math.pow(x_vector[i], 2) * _f.weight_vector[i]
				
			print 'x: %f' % x_vector[1]
			print 'y: %f' % x_vector[2]
			print 'num: %f' % num
			
			if num > 0:
				return 1
			return -1
			
		_f.weight_vector = weight_vector
		return _f
		
	@staticmethod
	def binary_classifier_error_function():
		def _binary_classifier_error_function(hypothesized_y, target_y):
			if hypothesized_y != target_y:
				return 1
			else:
				return 0
		return _binary_classifier_error_function
	
	@staticmethod
	def linear_regression_learning_function():
		def _linear_regression_learning_function(x_data, y_data, hypothesized_function, error_function):
			X = matrix(x_data)
			y = matrix(y_data)
			tmp = linalg.pinv(X) * y.transpose()	
			for i in range(len(tmp)):
				hypothesized_function.weight_vector[i] = tmp[i,0]
		return _linear_regression_learning_function
		
		
	@staticmethod
	def update_weights(weight_vector, x_vector, y_value):
		for i in range(len(weight_vector)):
			weight_vector[i] = weight_vector[i] + x_vector[i] * y_value
		return weight_vector
	
	@staticmethod
	def perceptron_learning_function(x_data, y_data, hypothesized_function, error_function):
		learned = False
		i = 0
		while not learned:
			misclassified = MachineLearning.misclassified_data(x_data, y_data, hypothesized_function, error_function)
			if len(misclassified) > 0:
				i+=1
				print "num misclassified: %d" % len(misclassified)
				random_index = int(random.random() * len(misclassified))
				hypothesized_function.weight_vector = MachineLearning.update_weights(hypothesized_function.weight_vector, misclassified[random_index], y_data[x_data.index(misclassified[random_index])])
				
				print "adjusting weights: %d" % i
			else:
				print "LEARNED"
				learned = True
		
		e = MachineLearning.in_sample_error(x_data, y_data, hypothesized_function, error_function)
		print 'e: %f' % e	
		
		
		
class MachineLearning:
	def __init__(self, *args, **kwargs):

		self.training_set_size = kwargs.get('training_set_size', 1000)
		
		self.x_data = kwargs.get('x_data', DataGenerator.random_matrix([2,self.training_set_size]))
		
		#self.target_function = kwargs.get('target_function', FunctionGenerator.binary_classifier([1,1,1]))
		self.target_function = kwargs.get('target_function', FunctionGenerator.nonlinear_binary_classifier([-0.6,1,1]))
		
		self.y_data = kwargs.get('y_data', DataGenerator.y_data_from_x_data(self.x_data, self.target_function))
		
		self.noise = kwargs.get('noise', 0.1)
		DataGenerator.add_noise(self.y_data, self.noise)
		
		self.hypothesized_function = kwargs.get('hypothesized_function', FunctionGenerator.binary_classifier([-10,0,1000]))
		
		self.error_function = kwargs.get('error_function', FunctionGenerator.binary_classifier_error_function())
		
		e = MachineLearning.in_sample_error(self.x_data, self.y_data, self.hypothesized_function, self.error_function)
		print 'e: %f' % e
# 		MachineLearning.adjust_weights(self.x_data, self.y_data, self.hypothesized_function, self.error_function)
		
		self.learning_function = FunctionGenerator.linear_regression_learning_function()
		self.learn()
			
		e = MachineLearning.in_sample_error(self.x_data, self.y_data, self.hypothesized_function, self.error_function)
		print 'e: %f' % e
	
	
	def repeat(self, n_times):
		average_in_sample_error = 0.0
		for i in range(n_times):
			self.x_data = DataGenerator.random_matrix([2,self.training_set_size])
			self.y_data = DataGenerator.y_data_from_x_data(self.x_data, self.target_function)
			
			self.learn()
			e = MachineLearning.in_sample_error(self.x_data, self.y_data, self.hypothesized_function, self.error_function)
			print 'e: %f' % e
			average_in_sample_error = (average_in_sample_error * i + e) / (i+1)
			
		print 'average_in_sample_error: %f' % average_in_sample_error
			
			
	def learn(self):
		self.learning_function(self.x_data, self.y_data, self.hypothesized_function, self.error_function)
	
	@staticmethod
	def in_sample_error(x_data, y_data, hypothesized_function, error_function):
		summed_error = 0.0
		for i in range(len(x_data)):
			summed_error += error_function(hypothesized_function(x_data[i]), y_data[i])
		
		return summed_error / len(x_data)
	
	@staticmethod
	def misclassified_data(x_data, y_data, hypothesized_function, error_function):
		misclassified = []
		for i in range(len(x_data)):
			if error_function(hypothesized_function(x_data[i]), y_data[i]) > 0:
				misclassified.append(x_data[i])
		return misclassified
	
	@staticmethod
	def update_weights(weight_vector, x_vector, y_value):
		for i in range(len(weight_vector)):
			weight_vector[i] = weight_vector[i] + x_vector[i] * y_value
		return weight_vector
	
	@staticmethod
	def adjust_weights(x_data, y_data, hypothesized_function, error_function):
		learned = False
		i = 0
		while not learned:
			misclassified = MachineLearning.misclassified_data(x_data, y_data, hypothesized_function, error_function)
			if len(misclassified) > 0:
				i+=1
				print "num misclassified: %d" % len(misclassified)
				random_index = int(random.random() * len(misclassified))
				hypothesized_function.weight_vector = MachineLearning.update_weights(hypothesized_function.weight_vector, misclassified[random_index], y_data[x_data.index(misclassified[random_index])])
				
				print "adjusting weights: %d" % i
			else:
				print "LEARNED"
				learned = True
		
		e = MachineLearning.in_sample_error(x_data, y_data, hypothesized_function, error_function)
		print 'e: %f' % e	
	
	
	
	
class DataGenerator:
	NATURAL = 0
	REAL = 1
	
	def __init__(self):
		pass
	
	@staticmethod
	def y_data_from_x_data(x_data, target_function):
		y_data = []
		for i in range(len(x_data)):
			y_data.append(target_function(x_data[i]))
		return y_data
	
	@staticmethod
	def add_noise(y_data, noise):
		""" adds noise by flipping the sign of 'noise * 100' percent of
		y_data values """
		for i in range(len(y_data)):
			if random.random() < noise:
				y_data[i] = y_data[i] * -1
		
	
	@staticmethod
	def random_vector(number_of_dimensions=2, min=-1, max=1, type=REAL, include_threshold_multiplier=True):
		vector = []
		if include_threshold_multiplier:
			vector.append(1)
			
		for i in range(number_of_dimensions):
			random_number = random.random() * (max - min) + min
			if type == DataGenerator.REAL:
				vector.append(random_number)
			else:
				vector.append(int(random_number))
			
		return vector
	
	@staticmethod
	def random_matrix(shape=[2,100], min=-1, max=1, type=REAL, include_threshold_multiplier=True):
		"""generates a 2-D list representing a matrix

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    Args:
        shape: The shape of the desired matrix
        min: The minimum value of any given value in the matrix 
        	(defaults to real numbers between 0-1 by default). 
        	This is exclusive i.e to get a min integer of 1 use min=0.
        max: The max value of any given value in the matrix. 
        
    Returns:
        A list representing a random generated matrix
        
    Raises:
    """
		matrix = []
		for i in range(shape[1]):
			matrix.append(DataGenerator.random_vector(shape[0], min, max, type, include_threshold_multiplier))
			
		return matrix	
		
		

m = MachineLearning({'noise':0.1})

#m.repeat(1000)
x = [] 
y = [] 
x_2 = [] 
y_2 = []

for i in range(len(m.x_data)):
	if m.y_data[i] == 1:
		x.append(m.x_data[i][1])
		y.append(m.x_data[i][2])
	else:
		x_2.append(m.x_data[i][1])
		y_2.append(m.x_data[i][2])

plt.plot(x,y,'ro')
plt.plot(x_2,y_2,'bo')
plt.show()
