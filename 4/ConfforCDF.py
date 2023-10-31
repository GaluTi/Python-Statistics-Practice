import numpy as np
import scipy as sc
import scipy.stats as st
import statsmodels.distributions.empirical_distribution as mod
import matplotlib.pyplot as plt

n=100
#Генерируем нормальную выборку
Norm=np.random.normal(0,5,size=n)
funN=mod.ECDF(Norm)
plt.plot(funN.x,funN.y)

#Строим интервалы ДВК и Колмогорова
alpha=0.05
epsDVK = np.sqrt(-np.log(alpha/2)/(2*n))
#берем обратную к колмогоровской ф.р., но учитываем, что в python это kolmogorov это 1-F
epsKolm = sc.special.kolmogi(alpha)/np.sqrt(n)
#эти числа практически одинаковы при больших n, если только мы не возьмем совсем малое альфа
#Поэтому будем строить только DVK
funNlowD=funN.y-epsDVK
funNupD=funN.y+epsDVK
#Строим интервал поточечно
#ppf - квантиль
funNlowPoint = funN.y - st.norm.ppf(1-alpha/2)*np.sqrt(funN.y*(1-funN.y))/np.sqrt(n)
funNupPoint = funN.y + st.norm.ppf(1-alpha/2)*np.sqrt(funN.y*(1-funN.y))/np.sqrt(n)
fig=plt.figure()
plt.plot(funN.x,funNlowD)
plt.plot(funN.x,st.norm.cdf(funN.x,0,5))
plt.plot(funN.x,funNupD)
fig=plt.figure()
plt.plot(funN.x,funNupPoint)
plt.plot(funN.x,st.norm.cdf(funN.x,0,5))
plt.plot(funN.x,funNlowPoint)
#Интервалы из неравенства Дворецкого и теоремы Колмогорова идентичны
#интервал поточечный уже, но где-то да вылазит за пределы отрезка

#Генерируем выборку Коши методом обратной функции
Cauch = np.tan(2*np.pi*np.random.rand(n))
funC=mod.ECDF(Cauch)

#То же для Б
funClowD=funC.y-epsDVK
funCupD=funC.y+epsDVK
#Строим интервал поточечно
#ppf - квантиль
funClowPoint = funC.y - st.norm.ppf(1-alpha/2)*np.sqrt(funC.y*(1-funC.y))/np.sqrt(n)
funCupPoint = funC.y + st.norm.ppf(1-alpha/2)*np.sqrt(funC.y*(1-funC.y))/np.sqrt(n)
fig=plt.figure()
plt.plot(funC.x,funClowD)
plt.plot(funC.x,np.arctan(funC.x)/np.pi+1/2)
plt.plot(funC.x,funCupD)
fig=plt.figure()
plt.plot(funC.x,funCupPoint)
plt.plot(funC.x,np.arctan(funC.x)/np.pi+1/2)
plt.plot(funC.x,funClowPoint)
#На самом деле нет никакой разницы между Коши и нормальным, они оба непрерывны 
#и никаких неудобств Коши нам не создает

#Попробуем дискретную выборку
Bin=np.random.binomial(5,0.5,size=n)
funB=mod.ECDF(Bin)
plt.plot(funB.x,funB.y)

#Строим интервалы ДВК и Колмогорова
funBlowD=funB.y-epsDVK
funBupD=funB.y+epsDVK
#Строим интервал поточечно
#ppf - квантиль
funBlowPoint = funB.y - st.norm.ppf(1-alpha/2)*np.sqrt(funB.y*(1-funB.y))/np.sqrt(n)
funBupPoint = funB.y + st.norm.ppf(1-alpha/2)*np.sqrt(funB.y*(1-funB.y))/np.sqrt(n)
fig=plt.figure()
plt.plot(funB.x,funBlowD)
plt.plot(funB.x,st.binom.cdf(funB.x,5,0.5))
plt.plot(funB.x,funBupD)
#здесь важно только попали значения в точка 0,1,...5 между значениями верхней и нижней границы
fig=plt.figure()
plt.plot(funB.x,funBupPoint)
plt.plot(funB.x,st.binom.cdf(funB.x,5,0.5))
plt.plot(funB.x,funBlowPoint)


#Теперь испытаем наши нормальный и биномиальный интервалы на выборках
Iter = 500
pDVK,pPoint=np.zeros(Iter),np.zeros(Iter)

for i in range(Iter):
    Norm=np.random.normal(0,5,size=n)
    funN=mod.ECDF(Norm)
    funNlowD=funN.y-epsDVK
    funNupD=funN.y+epsDVK
    funNlowPoint = funN.y - st.norm.ppf(1-alpha/2)*np.sqrt(funN.y*(1-funN.y))/np.sqrt(n)
    funNupPoint = funN.y + st.norm.ppf(1-alpha/2)*np.sqrt(funN.y*(1-funN.y))/np.sqrt(n)
    #сделаем сдвинутый массив абсцисс
    x3 = np.delete(funN.x,0)
    x2=np.append(x3,float("inf"))
    #ищем максимальное расстояние между функциями распределения, пользуясь тем,
    #что оно либо ЭФР - ф.р. в точке, либо ф.р. в точке - ЭФР в предыдущей точке
    D=max(max(funN.y-st.norm.cdf(funN.x,0,5)),max(st.norm.cdf(x2,0,5)-funN.y))
    pDVK[i]=True
    pPoint[i]=True
    if (D>epsDVK):
            pDVK[i]=False
    #уберем -бесконечность и бесконечность из допустимых значений
    y3=np.delete(funN.y,0)
    x3=np.delete(x3,len(x3)-1)
    y3=np.delete(y3,len(y3)-1)
    Dpoint= max(np.sqrt(n)*abs(st.norm.cdf(x3,0,5)-y3)/np.sqrt(y3*(1-y3)))
    if (Dpoint> st.norm.ppf(1-alpha/2)):
            pPoint[i]=False
            
print(sum(pDVK)/Iter)
print(sum(pPoint)/Iter)
#ДВК справляется довольно точно, поточечный интервал не работает,
#что вполне предсказуемо - он стоит интервалы поточечно, а на всем промежутке не накрывает

#биномиальный
pDVK,pPoint=np.zeros(Iter),np.zeros(Iter)
for i in range(Iter):
    Bin=np.random.binomial(5,0.5,size=n)
    funB=mod.ECDF(Bin)  
    funBlowD=funB.y-epsDVK
    funBupD=funB.y+epsDVK
    funBlowPoint = funB.y - st.norm.ppf(1-alpha/2)*np.sqrt(funB.y*(1-funB.y))/np.sqrt(n)
    funBupPoint = funB.y + st.norm.ppf(1-alpha/2)*np.sqrt(funB.y*(1-funB.y))/np.sqrt(n)
    #сделаем сдвинутый массив абсцисс
    x3 = np.delete(funB.x,0)
    #ищем максимальное расстояние между функциями распределения
    D=max(funB.y-st.binom.cdf(funB.x,5,0.5))
    pDVK[i]=True
    pPoint[i]=True
    if (D>epsDVK):
            pDVK[i]=False
    #уберем -бесконечность и бесконечность из допустимых значений
    y3=np.delete(funB.y,0)
    x3=np.delete(x3,len(x3)-1)
    y3=np.delete(y3,len(y3)-1)
    Dpoint= max(np.sqrt(n)*abs(st.binom.cdf(x3,5,0.5)-y3)/np.sqrt(y3*(1-y3)))
    if (Dpoint> st.norm.ppf(1-alpha/2)):
            pPoint[i]=False
            
print(sum(pDVK)/Iter)
print(sum(pPoint)/Iter)
#Опять же ДВК сработал наверняка, даже слишком 
#(точность должна быть 5%, а у нас выше - значит мы берем излишне широкий интервал
#поточечный интервал не работает на всей оси.