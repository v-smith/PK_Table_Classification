import numpy as np

# f = w * x #function that is linear combination of weights and inputs (don't care about the bias here)

# f = 2 * x
X = np.array([1,2,3,4], dtype= np.float32)
y = np.array([2,4,6,8], dtype= np.float32)

#initiliase weights
w= 0.0

#model pred
def forward(x):
    return w * x
#loss
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()
#grad
#MSE = 1/N * (w*x-y)**2
#dJ/dw= 1/N 2x (w*x -y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

print( f'Pred before training: f(5) = {forward(5):.3f}')

#Training
learning_rate= 0.01
n_iters = 10

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)

    #loss
    l= loss(y, y_pred)

    #grads
    dw = gradient(X, y, y_pred)

    #update weights
    w -= learning_rate* dw

    if epoch % 1 ==0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print( f'Pred after training: f(5) = {forward(5):.3f}')
