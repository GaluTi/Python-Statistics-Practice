#Доверительные  интервалы для одномерной равномерной выборки
import numpy.random as rand
import numpy as np
pPerc, pPiv = np.zeros(1000), np.zeros(1000)
#истинное значение параметра theta
th=1
for j in range(1000):
    #моделируем выборку и оцениваем ее максимум
    x=rand.random(50)*th
    th1=max(x)  
    thX=np.zeros(100)
    for i in range(100):
        thX[i]=max(rand.random(50)*th1)
    pPerc[j]=0
    pPiv[j]=0
    #Если не попали в интервал, то кладем в счетчик 1.
    if np.mean(1*(thX>th))<0.025 or np.mean(1*(thX<th))<0.025:
          pPerc[j]=1
    if np.mean(1*((2*th1-thX)>th))<0.025 or np.mean(1*((2*th1-thX)<th))<0.025:
      pPiv[j]=1
#Выводим долю числа испытаний, когда интервал не накрыл параметр. По идее это должно быть где-то 0.05
sum(pPiv/1000)
sum(pPerc/1000)
#Percentile метод не сработал вообще, а вот Pivotal сработал более или менее

pPerc, pPiv = np.zeros(1000), np.zeros(1000)
#истинное значение параметра theta
th=1
for j in range(1000):
    #моделируем выборку и оцениваем ее максимум несмещенно
    x=rand.random(50)*th
    th1=max(x)*51/50  
    thX=np.zeros(100)
    for i in range(100):
        thX[i]=max(rand.random(50)*th1)*51/50
    pPerc[j]=0
    pPiv[j]=0
    #Если не попали в интервал, то кладем в счетчик 1.
    if np.mean(1*(thX>th))<0.025 or np.mean(1*(thX<th))<0.025:
          pPerc[j]=1
    if np.mean(1*(2*th1-thX>th))<0.025 or np.mean(1*(2*th1-thX<th))<0.025:
      pPiv[j]=1
#Выводим долю числа испытаний, когда интервал не накрыл параметр. По идее это должно быть где-то 0.05
sum(pPiv/1000)
sum(pPerc/1000)
#С Percentile методом стало лучше, но все равно не очень, Pivotal в порядке

pPerc, pPiv = np.zeros(1000), np.zeros(1000)
#истинное значение параметра theta
th=1
for j in range(1000):
    #моделируем нормальную выборку и оцениваем ее средним
    x=rand.normal(size=50)+th
    th1=np.mean(x)
    thX=np.zeros(100)
    for i in range(100):
        thX[i]=np.mean(rand.normal(size=50)+th1)
    pPerc[j]=0
    pPiv[j]=0
    #Если не попали в интервал, то кладем в счетчик 1.
    if np.mean(1*(thX>th))<0.025 or np.mean(1*(thX<th))<0.025:
          pPerc[j]=1
    if np.mean(1*(2*th1-thX>th))<0.025 or np.mean(1*(2*th1-thX<th))<0.025:
          pPiv[j]=1
#Выводим долю числа испытаний, когда интервал не накрыл параметр. По идее это должно быть где-то 0.05
sum(pPiv/1000)
sum(pPerc/1000)
#На симметричном распределении при отсутствии смещения все стало хорошо
