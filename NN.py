import numpy as np
import matplotlib.pyplot as plt

anglesBlue = np.random.uniform(0,2*np.pi,100)
radiusBlue = np.random.uniform(0,1,100)
# radiusBlue = 3
x1 = radiusBlue * np.cos(anglesBlue)
y1 = radiusBlue * np.sin(anglesBlue)

anglesRed = np.random.uniform(0,2*np.pi,50)
radiusRed = np.random.uniform(1,2,50)
x2 = radiusRed * np.cos(anglesRed)
y2 = radiusRed * np.sin(anglesRed)

fig , ax = plt.subplots(1,2)

ax[0].scatter(x1,y1,label="blue",c="blue")
ax[0].scatter(x2,y2,label="red",c="red")

ax[0].legend()

x1=np.concatenate((x1,x2))
y1=np.concatenate((y1,y2))
data = np.column_stack((x1,y1))
np.random.shuffle(data)
label = np.array([1 if np.sqrt(i**2+j**2) <=1 else 0 for i,j in data])


# plt.scatter(x1,y1)


#initialize parameters

def init():
    W1 = np.random.rand(2,2)
    B1 = np.zeros((2,1))

    W2 = np.random.rand(2,2)
    B2 = np.zeros((2,1))

    W3 = np.random.rand(1,2)
    B3 = np.zeros((1,1))

    parameters = { 'w1':W1,'b1':B1,'w2':W2,'b2':B2,'w3':W3,'b3':B3}
    print("the initial parameters are:",parameters)
    return parameters


#forward propogation

def fw_prop(x,parameters):
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']
    w3=parameters['w3']
    b3=parameters['b3']
    z1= np.dot(w1,x) +b1 # or np.dot(w1,x)  in numpy for matrix multiplication and dot product the same function can be used 🤡
    a1=np.tanh(z1)
    z2 = np.dot(w2,a1)+b2
    a2= np.tanh(z2)
    z3 = np.dot(w3,a2)+b3
    a3 = 1/(1+np.exp(-z3)) #sigmoid
    return {
        'z1':z1,
        'a1':a1,
        'z2':z2,
        'a2':a2,
        'z3':z3,
        'a3':a3
    }

def cost_fun(y, y_pred):
    m = y.shape[0]  # Number of examples
    epsilon = 1e-15  # Small value to avoid taking the log of zero

    # Clip predicted values to avoid numerical instability
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Binary cross-entropy formula
    cost = - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    print("the cost is:",cost)
    return cost

def back_prop(x,y,parameters,neuron_vals):
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']
    w3=parameters['w3']
    b3=parameters['b3']
    z1 = neuron_vals['z1']
    a1 = neuron_vals['a1']
    z2 = neuron_vals['z2']
    a2 = neuron_vals['a2']
    z3 = neuron_vals['z3']
    a3 = neuron_vals['a3']

    m = x.shape[1]
    
    dz3 = a3-y
    dw3 = (1/m)*(np.dot(dz3,a2.T))
    # db3 = dz3.mean(axis=1)
    db3 = np.mean(dz3,axis=1,keepdims=True)

    dz2 = np.dot(w3.T,dz3)*(1-np.tanh(z2)**2)
    dw2 = (1/m)*(np.dot(dz2,a1.T))
    # db2 = dz2.mean(axis=1)
    db2 = np.mean(dz2,axis=1,keepdims=True)

    dz1 = np.dot(w2.T,dz2)*(1-np.tanh(z1)**2)
    dw1 = (1/m)*(np.dot(dz1,x.T))
    # db1 = dz1.mean(axis=1)
    db1 = np.mean(dz1,axis=1,keepdims=True)
    
    grad = {
        'dw1':dw1,
        'db1':db1,
        'dw2':dw2,
        'db2':db2,
        'dw3':dw3,
        'db3':db3,  
    }
    print("the gradients are:",grad)

    return grad

def update_param(l_r,gradient,parameters):
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']
    w3=parameters['w3']
    b3=parameters['b3']
    
    w1-=l_r * gradient['dw1']
    b1-=l_r * gradient['db1']
    w2-=l_r * gradient['dw2']
    b2-=l_r * gradient['db2']
    w3-=l_r * gradient['dw3']
    b3-=l_r * gradient['db3']

    new_parameters = { 'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3}

    print("the updated parameters are:",new_parameters)

    return new_parameters

def model(x,y,l_r,epoch):
    parameters = init()
    cost=[]
    for i in range(epoch):
        fw_props = fw_prop(x,parameters)
        cost.append(cost_fun(y,fw_props['a3']))
        grad = back_prop(x,y,parameters,fw_props)
        parameters = update_param(l_r,grad,parameters)
    return parameters , cost

param,cst=model(data.T*data.T,label,0.3,200)
t=np.arange(0,200)
ax[1].plot(t,cst)



plt.show()