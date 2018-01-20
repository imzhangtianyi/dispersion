import numpy as np
import pandas as pd
from StringIO import StringIO
#==================================set features===================================================================
class extract_features:
    def __init__(self, filename):
        self.filename = filename
        a = open(filename)
        t = a.read()
        a.close()
        depth_index = t.find('DEPTH IN CM')
        t = t[depth_index:]
        zeta_index = t.find('ZETA')
        d = t[:zeta_index]
        d = d[:d.find('W')]
        t = t[zeta_index:]
        self.text = t
        self.W_index = t.find('W')
        self.depth = d
    
    def properties(self):
        return pd.read_csv(StringIO(self.text[self.W_index:]),sep='\s+')
    
    def zetas(self):
        zeta = pd.read_csv(StringIO(self.text[:self.W_index]), header=None, sep='\s+', skiprows=1)
        z = zeta.values
        zeta_val = z.ravel()
        zeta_val = zeta_val[~np.isnan(zeta_val)]
        # times of zeta crosses 0
        zeta0 = sum(np.diff(np.sign(zeta_val)) != 0)
        # times of dzeta crosses 0
        dz = np.gradient(zeta_val, 1)
        dz0 = sum(np.diff(np.sign(dz)) != 0)
        # shelf break
        de = pd.read_csv(StringIO(self.depth), header=None, sep='\s+', skiprows=1)
        depth_val = de.values.ravel()
        depth_val = depth_val[~np.isnan(depth_val)]
        depth_val = depth_val/depth_val.max()*np.fabs(zeta_val).max()
        depth_val_d = np.diff(depth_val)
        depth_val1 = depth_val_d[1:]
        depth_val1 = np.append(depth_val1, depth_val_d[-1])
        sb = list(depth_val1/depth_val_d > 3).index(1)
        zeta_val_d = np.diff(np.log(np.fabs(zeta_val)))
        wb = np.argmax(np.gradient(zeta_val_d,1)**2)
        dist = float(sb - wb)/len(depth_val)
        return pd.DataFrame(np.array([zeta0,dz0,abs(dist)]).reshape(1,3), columns=['zeta0','dz','wb'])