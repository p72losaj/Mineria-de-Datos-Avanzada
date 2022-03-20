import matplotlib.pyplot as plt # Graphics

# BAGGING'S CLASSIFIER

nElements = [750,3125,17070]
tests_bagging = [0.956, 0.883, 0.959]
trains_bagging = [0.971, 0.874,0.945]
errors_bagging = [0.044, 0.117,0.041];


# Test's graphics

plt.plot(nElements,tests_bagging,label = 'bgclassifier') # Creation of the rect
plt.xlabel('Number of elements of dataset') # Elements's dataset
plt.ylabel('Result of the test') # Test's results
plt.title('Relation Dataset and test') # Title
plt.legend() # Show the legend
plt.show() # Show the graphic

# Train's graphics

plt.plot(nElements,trains_bagging,label='bgclassifier') # Creation of the rect
plt.xlabel('Number of elements of dataset') # Elements's dataset
plt.ylabel('Result of the train') # Train's results
plt.title('Relation Dataset and train') # title
plt.legend() # Show the legend
plt.show() # Show the graphic

# Error's graphics

plt.plot(nElements,errors_bagging,label='bgerror') # Creation of the rect
plt.xlabel('Number of elements of dataset') # Elements's dataset
plt.ylabel('Result of the error') # Error's results
plt.title('Relation Dataset and error') # title
plt.legend() # Show the legend
plt.show() # Show the graphic
