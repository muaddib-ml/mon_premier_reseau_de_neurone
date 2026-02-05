import random
# lallallalalalalalallalala Manthra :P
def erreur(y_pred, y_true):
    return (y_pred - y_true)**2
    
def leaky_relu(x):
    return x if x > 0 else 0.01*x
    
w1 = [random.uniform(-1, 1) for _ in range(3)]
b1 = [random.uniform(-1, 1) for _ in range(3)]
w2 = [random.uniform(-1, 1) for _ in range(3)]
b2 = random.uniform(-1, 1)

x = 2.0
y_true = 5.0
lr = 0.001
epsilon = 1e-5

for iteration in range(10000):
    a1 = [leaky_relu(w1[i]*x + b1[i]) for i in range (3)]
    y_pred = sum([a1[i] * w2[i] for i in range(3)]) + b2
    loss = erreur(y_pred, y_true)

    grads_w2 = []
    for j in range(3):
        w_temp = w2[j] + epsilon
        y_temp = sum([a1[k] * (w_temp if k==j else w2[k]) for k in range(3)]) + b2
        loss_temp = erreur(y_temp, y_true)
        grad = (loss_temp - loss) / epsilon
        grads_w2.append(grad)

    for j in range(3):
        w2[j] -= lr * grads_w2[j]

    if iteration % 100 == 0:
        print(f"Iteration {iteration} | y_pred = {y_pred:.4f} | loss = {loss:.4f}")

