import numpy as np

# normalize input seismic data beofre processing 
def normalize_data (input_data):
  normalized_seismic_data = (input_data-np.mean(input_data))/np.std(input_data)  
  return normalized_seismic_data


# calculate seismic attribut on original seismic data
def calculate_seismic_attribute (seismic_cube, m1, m2, m3, rms_window = 3):
  arms = []
  for i in range(seismic_cube.shape[0]):
      for j in range(seismic_cube.shape[1]):
          a = np.square(seismic_cube[i,j])
          mask = np.ones(rms_window)/rms_window
          RMS_amp = np.sqrt(np.convolve(a, mask, 'same'))
          arms.append(RMS_amp)
  ARMS = np.array(arms)
  ARMS = np.reshape(ARMS, (m1, m2, m3))
  gx = list(ARMS)

  return gx
# interpret faults on seismic data 
def predict_faults(gx, m1, m2, m3, loaded_model, overlap = 12):
  overlap = 12 #overlap 
  n1, n2, n3 = 128, 128, 128
  c1 = np.round((m1+overlap)/(n1-overlap)+0.5)
  c2 = np.round((m2+overlap)/(n2-overlap)+0.5)
  c3 = np.round((m3+overlap)/(n3-overlap)+0.5)
  c1 = int(c1)
  c2 = int(c2)
  c3 = int(c3)
  p1 = (n1-overlap)*c1+overlap
  p2 = (n2-overlap)*c2+overlap
  p3 = (n3-overlap)*c3+overlap
  gx = np.reshape(gx,(m1,m2,m3))
  gp = np.zeros((p1,p2,p3),dtype=np.single)
  gy = np.zeros((p1,p2,p3),dtype=np.single)
  mk = np.zeros((p1,p2,p3),dtype=np.single)
  gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
  gp[0:m1,0:m2,0:m3]=gx
  sc = getMask(overlap)
  for k1 in range(c1):
      for k2 in range(c2):
          for k3 in range(c3):
              b1 = k1*n1-k1*overlap
              e1 = b1+n1
              b2 = k2*n2-k2*overlap
              e2 = b2+n2
              b3 = k3*n3-k3*overlap
              e3 = b3+n3
              gs[0,:,:,:,0]=gp[b1:e1,b2:e2,b3:e3]
              Y = loaded_model.predict(gs,verbose=1)
              Y = np.array(Y)
              gy[b1:e1,b2:e2,b3:e3]= gy[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
              mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
  gy = gy/mk
  gy = gy[0:m1,0:m2,0:m3]
  return gy

# set gaussian weights in the overlap bounaries  
def getMask(os, n1=128, n2=128, n3=128):
    sc = np.zeros((n1,n2,n3),dtype=np.single)
    sc = sc+1
    sp = np.zeros((os),dtype=np.single)
    sig = os/4
    sig = 0.5/(sig*sig)
    for ks in range(os):
        ds = ks-os+1
        sp[ks] = np.exp(-ds*ds*sig)
    for k1 in range(os):
        for k2 in range(n2):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k1]
                sc[n1-k1-1][k2][k3]=sp[k1]
    for k1 in range(n1):
        for k2 in range(os):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k2]
                sc[k1][n3-k2-1][k3]=sp[k2]
    for k1 in range(n1):
        for k2 in range(n2):
            for k3 in range(os):
                sc[k1][k2][k3]=sp[k3]
                sc[k1][k2][n3-k3-1]=sp[k3]
    return sc

def denoize_results(gy, trashhold = 0.95):    
  denoize = np.zeros(gy.shape)
  denoize[gy >= trashhold] = 1
  denoize[gy < trashhold] = np.nan
  return denoize


