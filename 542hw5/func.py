fig = plt.figure()
# %matplotlib notebook
# %matplotlib inline
# fig, ((ax1, ax2))= plt.subplots(1,2,figsize = (15,5))
# plt.figure()
x, y = 1, -1
beta0_axis = [beta0s[i] for i in range(len(beta0s)) for j in range(len(beta1s))]
beta1_axis = [beta1s[j] for i in range(len(beta0s)) for j in range(len(beta1s))]
# ll_func = [sigmoid( beta0s[i] + beta1s[j] * x) for i in range(len(beta0s)) for j in range(len(beta1s))]
ll_func = [log_likelihood(x,y,beta0s[i],beta1s[j]) for i in range(len(beta0s)) for j in range(len(beta1s))]
# ax1 = fig.add_subplot(111, projection='3d')
ax1 = fig.gca(projection='3d')
ax1.scatter(beta0_axis, beta1_axis, ll_func) # , label='x=1, y=-1'
ax1.set_xlabel('${\\beta_0}$')
ax1.set_ylabel('${\\beta_1}$')
ax1.set_xticks([-2,-1,0,1,2])
ax1.set_yticks([-2,-1,0,1,2])
ax1.set_zlabel('Log Likelihood')
ax1.legend()

# x, y = 1, 1
# beta0_axis = [beta0s[i] for i in range(len(beta0s)) for j in range(len(beta1s))]
# beta1_axis = [beta1s[j] for i in range(len(beta0s)) for j in range(len(beta1s))]
# ll_func = [log_likelihood(x,y,beta0s[i],beta1s[j]) for i in range(len(beta0s)) for j in range(len(beta1s))]
# # ll_func = [log_likelihood(x,y,beta0s[i],beta1s[i]) for i in range(len(beta0s))]
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot(beta0_axis, beta1_axis, ll_func, label='x=1, y=1')
# ax2.set_xlabel('${\\beta_0}$')
# ax2.set_ylabel('${\\beta_1}$')
# ax2.set_xticks([-2,-1,0,1,2])
# ax2.set_yticks([-2,-1,0,1,2])
# ax2.set_zlabel('Log Likelihood')
# ax2.legend()
plt.tight_layout()
plt.show()



def compute_loss(data, labels, beta_, beta0_):
  m = data.shape[0]
  # power = - labels @ (beta0_ + beta_ @ data.T)
  # return 1/m * np.sum(np.log( 1 + np.exp(power) ))
  total = 0
  for i in range(m):
    total += np.log(1 + np.exp(-labels[i] * (beta0_ + beta_@data[i])))
  return 1/m * total
def compute_gradient(data, labels, beta_, beta0_):
  m = data.shape[0]
  # power = - labels @ (beta0_ + beta_ @ data.T)
  # pwer1 = - (labels @ (beta0_ + beta_ @ data.T))
  # tmp = np.exp(power)@labels
  # dB_0 = -1/m * np.sum(np.exp(power)@labels / (1+np.exp(power1)))
  # dB = -1/m * np.sum(np.exp(power)@labels @ data / (1+np.exp(power1)))
  totalb, totalb0 = 0, 0
  for i in range(m):
    numer = np.exp(-labels[i] * (beta0_ + beta_@data[i]))
    denom = np.exp(- (labels[i] * (beta0_ + beta_@data[i])))
    totalb0 += (numer / (1 + denom)) * labels[i]
    totalb += (numer / (1 + denom)) * labels[i] * data[i]
  dB = -1/m * totalb
  dB_0 = -1/m * totalb0
  return dB, dB_0

def predict(X_test, y_test, beta_, beta0_):
  tmp = beta0_ + beta_ @ X_test.T
  sigmoid_output = 1/(1 + np.exp(-tmp))
  # print(sigmoid_output.shape)
  pred = [1 if p>=0.5 else -1 for p in sigmoid_output]
  y_pred = np.array(pred)
  cnt = 0
  for i in range(len(y_pred)):
    if y_test[i] == y_pred[i]:
      cnt+=1
  return cnt/len(y_pred)