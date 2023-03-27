import numpy as np
from gekko import GEKKO
import pandas as pd
from sklearn import datasets

data= datasets.load_boston()

xm1=pd.DataFrame(data.target, columns=["AGE"])
xm2=pd.DataFrame(data.target, columns=["RM"])
#xm3=pd.DataFrame(data.target, columns=["DIS"])
#xm4=pd.DataFrame(data.target, columns=["CRIM"])
#xm5=pd.DataFrame(data.target, columns=["TAX"])
#xm6=pd.DataFrame(data.target, columns=["RAD"])

ym=pd.DataFrame(data.target, columns=["MEDV"])

##y=a*(x1**b)*(x2**c)*(x3**d)*(x4**e)*(x5**f)*(x6**g)
m=GEKKO()

a=m.FV(value=0.1)
b=m.FV(value=0.1)
c=m.FV(value=0.1)
#d=m.FV(lb=-100.0 , ub=100.0)
#e=m.FV(lb=-100.0 , ub=100.0)
#f=m.FV(lb=-100.0 , ub=100.0)
#g=m.FV(lb=-100.0 , ub=100.0)

x1=m.Param(value=xm1)
x2=m.Param(value=xm2)
#x3=m.Param(value=xm3)
#x4=m.Param(value=xm4)
#x5=m.Param(value=xm5)
#x6=m.Param(value=xm6)

y=m.CV(value=ym)
y.FSTATUS=1

##
a.STATUS = 1
b.STATUS = 1
c.STATUS = 1
#d.STATUS = 1
#e.STATUS = 1
#f.STATUS = 1
#g.STATUS = 1

m.Equation(y==a*(x1**b)*(x2**c))#*(x3**d)*(x4**e)*(x5**f)*(x6**g))

m.options.IMODE=2
#m.options.SOLVER=1

m.solve()

print('a: ',a.value[0])
print('b: ',b.value[0])
print('c: ',c.value[0])
#print('d: ',d.value[0])
#print('e: ',e.value[0])
#print('f: ',f.value[0])
#print('g: ',g.value[0])
