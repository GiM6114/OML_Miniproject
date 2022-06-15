import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import time
import plotly.express as px
import numpy as np
from tqdm import tqdm
from plotly.subplots import make_subplots
import torch
import pickle

data = pd.read_csv('/Users/bessette/Desktop/OML_Miniproject-main/data/communities.data', header=None)
data = data.drop(labels=range(5), axis=1) # drop first 5 attributes (non predictive)
data = data.replace('?',np.nan)
data = data.dropna()

A = data.iloc[:,:].to_numpy(dtype='float64')
A[:,-1] = 1 # fixed input for each example for the bias term
b = data.iloc[:,-1].to_numpy(dtype='float64') # values to predict


max_iters = 200000
n_prec = 5

# This class stores the iteration indexes and execution time
# at which an optimization algorithm manages to have an error 
# lower than the 'precisions' values given. 
class PrecisionHolder:
    def __init__(self, real_value, precisions=[0.1, 0.05, 0.01, 0.005, 0.001]):
        self.real_value = real_value
        self.precisions = precisions
        self.n = len(precisions)
        self.precisions_itr = -np.ones(self.n, dtype='int64')
        self.precisions_tim = -np.ones(self.n, dtype='float64')
        self.ptr = 0 # keep track of which value has to be reached now
        
        
    def notifyValue(self, value, itr, time):
        error = abs(value - self.real_value)
        for i in range(self.n):
            if error < self.precisions[i]:
                if self.precisions_itr[i] < 0:
                    self.precisions_itr[i] = itr + 1
                    self.precisions_tim[i] = time
                    if self.ptr == self.n-1: # this means we reached all values needed
                        self.ptr = -1
                    else:
                        self.ptr += 1
            else:
                break
    
    
    def __str__(self):
        txt = ''
        for precision,itr in zip(self.precisions,self.precisions_itr):
            if itr >= 0:
                txt += f'Precision of {precision} reached after {itr} iterations.\n'
            else:
                txt += f'Precision of {precision} not reached.\n'
        return txt
    
    
    def allValuesAcquired(self):
        return self.ptr < 0


#Computes smoothness constant L for f
def calculate_L(A):
    eig = np.linalg.eigvals(A.T.dot(A))
    L = max(eig)/(A.shape[0])    
    return 2*L


def train_bgd_reg_lin(A, p_holder, max_iters=500000, verbose=False):
    x = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True)

    # define the model
    def forward(x,A):
        return A@x

    loss = torch.nn.MSELoss()
    learning_rate = 1 / calculate_L(A)
    optimizer = torch.optim.SGD(params=[x], lr=learning_rate)

    A_t = torch.tensor(A, dtype=torch.float64)
    b_t = torch.tensor(b, dtype=torch.float64)

    start_time = time.time()
    for n_iter in range(max_iters):
        b_pred = forward(x,A_t)
        l = loss(b_t, b_pred)
        p_holder.notifyValue(l, n_iter, time.time()-start_time)
        if p_holder.allValuesAcquired():
            break
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose and n_iter%10000==9999 and n_iter!=0:
            print(f'Loss for iteration {n_iter}/{max_iters-1} : {l.item()}')

    if verbose:
        print(f'Execution time : ', time.time()-start_time)


itr_gd = np.zeros([A.shape[1]-90,n_prec],dtype='int64')
# itr_gd[i,j] : nb iterations to reach precision 10^(1-j) with i dimensions
tim_gd = np.zeros([A.shape[1]-90,n_prec],dtype='float64')
for i in tqdm(range(A.shape[1]-90), desc="Nb of dimensions"):
    A_temp = A[:,:i+1]
    lowest_x = np.linalg.inv(A_temp.T@A_temp)@A_temp.T@b
    lowest_loss = np.sum(np.square(A_temp@lowest_x-b))/A_temp.shape[0]
    p_holder_gd = PrecisionHolder(real_value=lowest_loss)
    train_bgd_reg_lin(A_temp, p_holder_gd)
    itr_gd[i,:] = p_holder_gd.precisions_itr
    tim_gd[i,:] = p_holder_gd.precisions_tim



# Zero Order SGD
# f : function to minimize
# d : number of dimension of input
# max_iters : number of iterations to do

def ZO_SGD(f, f_augm, d, p_holder, m=1000, mean =0, std = 0.5, delta=0.1, eta=0.1):

    x = np.zeros(d)

    m_t = np.zeros(d)

    start_time = time.time()

    f_x = f(x)
    for n_iter in range(max_iters):
        grad_k = 0
        x_temp = np.tile(x.copy(),(m,1))
        u_j = np.random.normal(mean, std, (m, np.size(x)))
        x_temp += delta*u_j 
        f_x_du = np.expand_dims(f_augm(x_temp), axis=0)
        grad_k = np.mean(np.multiply((np.subtract(f_x_du.T, f_x))/delta,u_j), axis=0)
        x -= eta*grad_k
        f_x = f(x)
        p_holder.notifyValue(f_x, n_iter, time.time()-start_time)
        if p_holder.allValuesAcquired():
            break

def ZO_SGD_mse(A, b, p_holder):
    return ZO_SGD(lambda x: np.sum(np.square(A@x-b))/A.shape[0], lambda x: np.sum(np.square(np.matmul(A,x.T).T-b), axis=1)/A.shape[0], A.shape[1], p_holder)

# Plot Results
def plot_result(itr_zo, itr_gd):
    fig = make_subplots(rows=3, cols=2)
    for i in range(3):
        for j in range(2):
            if i*2+j >= 5:
                break
            fig.add_trace(go.Scatter(x=np.arange(1,np.shape(itr_gd)[0]+1), y=np.arange(1,np.shape(itr_gd)[0]+1), marker_color='rgba(0,0,0,255)'), row=i+1,col=j+1)
            x_list = np.arange(1,np.shape(itr_gd)[0]+1)
            y_list = itr_zo[:,i*2+j]/itr_gd[:,i*2+j]
            color_list = [0 if x > 0 else 1 for x in itr_zo[:,i*2+j]/itr_gd[:,i*2+j]]
            for tn in range(np.shape(itr_gd)[0]):
                fig.add_trace(
                    go.Scatter(
                        x=x_list[tn:tn+2],
                        y=y_list[tn:tn+2],
                        line_color=px.colors.qualitative.Plotly[color_list[tn]],
                        mode='lines'
                    ), row=i+1, col=j+1
                )
    #fig.update_layout(xaxis_title='Number of dimensions', yaxis_title='Number of iterations', title='Number of iterations necessary to reach accuracy of 5*10^-2 with respect to the number of dimensions', showlegend=False)
    fig.update_layout(showlegend=False)
    fig.add_trace(go.Scatter())
    fig.show()


itr_ZO_SGD = np.zeros([A.shape[1]-90,n_prec],dtype='int64')
# itr_gd[i,j] : nb iterations to reach precision 10^(1-j) with i dimensions
tim_ZO_SGD = np.zeros([A.shape[1]-90,n_prec],dtype='float64')
for i in tqdm(range(A.shape[1]-90), desc="Nb of dimensions"):
    A_temp = A[:,:i+1]
    lowest_x = np.linalg.inv(A_temp.T@A_temp)@A_temp.T@b
    lowest_loss = np.sum(np.square(A_temp@lowest_x-b))/A_temp.shape[0]
    p_holder_ZO_SGD = PrecisionHolder(real_value=lowest_loss)
    ZO_SGD_mse(A_temp,b, p_holder_ZO_SGD)
    itr_ZO_SGD[i,:] = p_holder_ZO_SGD.precisions_itr
    tim_ZO_SGD[i,:] = p_holder_ZO_SGD.precisions_tim


plot_result(itr_ZO_SGD,itr_gd)


print(itr_ZO_SGD)
print(itr_gd)















