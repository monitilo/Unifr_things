# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:50:12 2019

@author: ChiarelG
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

# %%
""" copypastie la data que sale del cst a un txt. Para las curvas de cada NP """


for NP in ["noNP", 1, 3, 5,"plot"]:
    print(NP)


#NP = "noNP"
if NP == "noNP":
#    data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/No NP dimer.txt',dtype=str, delimiter=" \t ")
    data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/no NP monomer.txt',dtype=str, delimiter=" \t ")

elif NP == 1:
#    data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/1nm NP Dimer.txt',dtype=str, delimiter=" \t ")
    data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/1nm NP monomer.txt',dtype=str, delimiter=" \t ")

elif NP == 3:
#    data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/3nm NP Dimer.txt',dtype=str, delimiter=" \t ")
    data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/3nm NP monomer.txt',dtype=str, delimiter=" \t ")

elif NP ==5:
#    data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/5nm NP Dimer.txt',dtype=str, delimiter=" \t ")
    data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/5nm NP monomer.txt',dtype=str, delimiter=" \t ")

dist = {}
field = {}

p=0
dist[p]=[]
field[p]=[]
i=16
wav = []
wav.append(float(data[4].split()[2].replace('_',' ').split()[0]))
for j in range(len(data)):
    try:
        try:
            dist[p].append(float(data[i].split()[0].replace(',','.')))
            field[p].append(float(data[i].split()[1].replace(',','.')))
            i+=1
    #        print(i, data[i],"b")
        except:
            print(i, data[i],"aaa", "p", p)
            wav.append(float(data[i].split()[2].replace('_',' ').split()[0]))
            i+=12
            p+=1
            dist[p]=[]
            field[p]=[]
    except:
        print(i, len(data))
        break

print("wavelenght:", len(wav), "files:", len(dist), len(field), "points:", len(dist[1]), len(field[1]))

step = (len(dist[1])-1)/max(dist[1])

"""El centro esta en 8, dondo centre la NP que mide 0.5, 1.5, o 2.5 de radio
El Rod esta a -5 nm del centro (en 3) """
start = int(step*3.8)
end = int(step*4.5)
for i in range(42,43): #len(dist)):
    plt.plot(dist[i][start:end], field[i][start:end], 'r', lw=4)
    plt.plot(dist[i],field[i])

maxing = []
for i in range(len(field)):
    maxing.append(field[i][start])

resonance = np.where(np.max(maxing)==maxing)[0][0]
print(resonance, "resonance")

#%%
if NP == 1:
    dist1nm = dist[resonance]
    field1nm=field[resonance]
elif NP == 3:
    dist3nm = dist[resonance]
    field3nm=field[resonance]
elif NP ==5:
    dist5nm = dist[resonance]
    field5nm = field[resonance]
elif NP == "noNP":
    distnoNP = dist[resonance]
    fieldnoNP = field[resonance]
else:
    plt.plot(dist1nm, field1nm)
    plt.plot(dist3nm, field3nm)
    plt.plot(dist5nm, field5nm)
    plt.plot(distnoNP, fieldnoNP)
    plt.plot(dist5nm[start:end+1], field5nm[start:end+1], 'm', lw=2)
    plt.vlines(dist5nm[start],0,field5nm[start], 'm',alpha=0.5)
    plt.vlines(dist5nm[end],0,field5nm[end], 'm',alpha=0.5)
    plt.xlim(0,16)
    plt.ylim(bottom=0)
    plt.ylabel("E/$E_0$")
    plt.xlabel("Distance (nm)")
# %%
Ef = []
start = int(step*3.8)
end = int(step*4.5)
for i in range(len(dist)):
    Ef.append(np.mean(field[i][start:end]))

if NP == 1:
    Ef1nm = Ef
elif NP == 3:
    Ef3nm = Ef
elif NP ==5:
    Ef5nm = Ef
elif NP =="noNP":
    EfnoNP = Ef
    wavnoNP = wav
    labels= ["noNP"]
    #plt.plot(wav,Ef, '-*')
    plt.plot(wav,EfnoNP/max(EfnoNP),'-*')
    plt.xlim(775,792)
    plt.ylim(0.99,1.001)
    plt.legend(labels)
#    plt.plot(wav,EfnoNP, '-')
#    plt.xlim(700,900)
#    plt.ylim(0,150)
    plt.ylabel("Normalized E field")
    plt.xlabel("wavelength (nm)")
    plt.vlines(wav[np.where(np.max(EfnoNP)==EfnoNP)[0][0]],0,1.1,color='k',alpha=0.5)
    plt.text(wav[np.where(np.max(EfnoNP)==EfnoNP)[0][0]]-1,1.0003,wav[np.where(np.max(EfnoNP)==EfnoNP)[0][0]])
    print(max(EfnoNP))

else:
    labels= ["1nm", "3nm", "5nm","no NP"]
    #plt.plot(wav,Ef, '-*')
#    plt.plot(wav,Ef1nm/max(Ef1nm), '-*', wav, Ef3nm/max(Ef3nm), '-*', wav,Ef5nm/max(Ef5nm), '-*', wavnoNP,EfnoNP/max(EfnoNP), '-*')
#    plt.xlim(775,792)
#    plt.xlim(725,731)
#    plt.ylim(0.995,1.001)
    plt.plot(wav,Ef1nm, '-', wav, Ef3nm, '-', wav,Ef5nm, '-', wavnoNP,EfnoNP, '-')
    plt.legend(labels)
#    plt.xlim(700,900)
#    plt.ylim(0,150)
    plt.ylabel("Normalized E field")
    plt.xlabel("wavelength (nm)")
#    plt.vlines(wav[np.where(np.max(Ef1nm)==Ef1nm)[0][0]],0,1.1,color='k',alpha=0.5)
#    plt.vlines(wav[np.where(np.max(Ef3nm)==Ef3nm)[0][0]],0,1.1,color='k',alpha=0.5)
    plt.vlines(wav[np.where(np.max(Ef5nm)==Ef5nm)[0][0]],0,1.1,color='k',alpha=0.5)
    plt.vlines(wavnoNP[np.where(np.max(EfnoNP)==EfnoNP)[0][0]],0,1.1,color='k',alpha=0.5)
#    plt.text(wav[np.where(np.max(Ef1nm)==Ef1nm)[0][0]]-1,1.0003,wav[np.where(np.max(Ef1nm)==Ef1nm)[0][0]])
#    plt.text(wav[np.where(np.max(Ef3nm)==Ef3nm)[0][0]],1.0003,wav[np.where(np.max(Ef3nm)==Ef3nm)[0][0]])
#    plt.text(wav[np.where(np.max(Ef5nm)==Ef5nm)[0][0]],1.0003,wav[np.where(np.max(Ef5nm)==Ef5nm)[0][0]])
#    plt.text(wavnoNP[np.where(np.max(EfnoNP)==EfnoNP)[0][0]]-0.5,1.0003,wavnoNP[np.where(np.max(EfnoNP)==EfnoNP)[0][0]])
    
    print(max(Ef1nm), max(Ef3nm), max(Ef5nm), max(EfnoNP))
#plt.plot(wav,Ef3nm, '-')
#plt.plot(wav,Ef5nm, '-')



# %% For the cross sections

#data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/ACS NPs.txt',dtype=str, delimiter=" \t ")
data = np.loadtxt('C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/monomer ACS.txt',dtype=str, delimiter=" \t ")



wave = {}
ACS = {}

p=0
wave[p]=[]
ACS[p]=[]
i=17
sizenp = []
sizenp.append(str(data[4].split()[4][-7:-1]))
for j in range(len(data)):
    try:
        try:
            wave[p].append(float(data[i].split()[0].replace(',','.')))
            ACS[p].append(float(data[i].split()[1].replace(',','.')))
            i+=1
    #        print(i, data[i],"b")
        except:
            print(i, data[i],"aaa", "p", p)
            sizenp.append(str(data[i].split()[4][-7:-1]))
            i+=13
            p+=1
            wave[p]=[]
            ACS[p]=[]
    except:
        print(i, len(data))
        break

#print("sizenp:", len(sizenp), "files:", len(wave), len(ACS), "points:", len(wave[1]), len(ACS[1]))

print(sizenp)
for i in range(len(wave)): plt.plot(wave[i], ACS[i]/np.max(ACS[i]), label=sizenp[i]),plt.legend()

# %%
labels= ["1nm", "3nm", "5nm"]
texto = [1,0,0]
for i in range(len(wave)): 
    plt.plot(wave[i], ACS[i]/np.max(ACS[i]))
    plt.vlines(wave[i][np.where(np.max(ACS[i])==ACS[i])[0][0]],0,1.1,color='k',alpha=0.5)
#    plt.text(wave[i][np.where(np.max(ACS[i])==ACS[i])[0][0]]-texto[i],1.0003,wave[i][np.where(np.max(ACS[i])==ACS[i])[0][0]])

#plt.legend(labels)
plt.ylabel("Normalized ACS")
plt.xlabel("wavelength (nm)")
#plt.xlim(775,792)
plt.xlim(723,728)
plt.ylim(0.99,1.001)


# %%
#dataname = 'C:/Users/chiarelG/Downloads/Projects/Local Surface Plasmon Resonance/All_curves.txt'
"""cargar la data a mano"""
#data = np.loadtxt(dataname)

#text_dist = All_curvestxt[:,0]
#text_Ef = All_curvestxt[:,1]
#text_dist,text_Ef, lambda_text = All_curvestxt[:,0], All_curvestxt[:,1], All_curvestxt[:,4]
text_dist,text_Ef, lambda_text = All_curves[:,0], All_curves[:,1], All_curves[:,4]
#text_dist,text_Ef, lambda_text = All_curves1nm[:,0], All_curves1nm[:,1], All_curves1nm[:,4]


#%%

dist = np.zeros((100, len(text_dist[:])))
Ef = np.zeros((100, len(text_Ef[:])))
lambdas = []
p=0
j=0
Es = []
b=0

for i in range(len(text_dist)):
    try:

        dist[j, p] = float(text_dist[i])+1
        Ef[j, p] = np.float32(text_Ef[i])
        Es.append(float(text_Ef[i]))
        p+=1
    except:
        b+=1
#        if j< 200:
#            print("j=",j, "p=",p , "i=",i)
#        print(text_dist[i], text_Ef[i], lambda_text[i])
        if lambda_text[i] != 'nan':
#            print(lambda_text[i])
            lambdas.append(float(lambda_text[i][:-5]))
        j+=1

easy = np.array(np.split(np.array(Es),len(lambdas)))

print("b =", b)

aux =[]
auxw =[]
for i in range(len(Ef[:,0])):
        aux.append(np.array(np.trim_zeros(Ef[i,:])))
        auxw.append(np.array(np.trim_zeros(dist[i,:])))


# %%

#plt.plot(np.trim_zeros(Ef[1,:]))

for i in range(len(aux)):
    plt.plot(auxw[i]-1, aux[i])


#print(auxw[2]-1)
print("\n")
print(len(auxw[2]-1), auxw[2][0]-1, auxw[2][-1]-1)
print(lambdas, "\n", len(lambdas))
#%%

#curves = np.array(aux).T
#
#algo = np.zeros((len(curves), len(curves[2])))
#for i in range(len(curves)):
#    for j in range(len(curves[2])):
##        print(i,j)
#        try:
#            algo[i,j] = np.nan_to_num(curves[i][j])
#        except:
#            pass
#
#aa= algo.T
#
#savedata = []
#for i in range(len(aa[0,:])):
#    if np.sum(aa[:,i]) != 0:
#        savedata.append(aa[:,i])

print("total curves", len(easy), "\n total lambdas", len(lambdas))
print(easy.shape)
#savelambda = np.zeros((1, len(lambdas[:])))
#for i in range(len(lambdas)):
#    savelambda[0,i] = lambdas[i]
#otro = []
#for i in range(len(curves)):
#    print(sum(np.nan_to_num(curves[i])))
#    if np.sum(np.nan_to_num(curves[i])) != 0:
#        otro.append(np.trim_zeros(np.nan_to_num(curves[i])))


np.savetxt("Curvas_para_analisis_5nm2.txt", easy.T, delimiter="    ", newline='\r\n', fmt='%.4e')
np.savetxt("Curvas_para_analisis-lambdas_5nm2.txt", np.array(lambdas).ravel())

#%%


import numpy as np
import matplotlib.pyplot as plt

N=100

trace = np.random.normal(0, size=N)*10
trace[-8] = 60
for i in range(50):
    trace[i] = 60 + trace[i]

old_threshold = 1
converge = False
#while converge == False:
distance = []
step = 1
for i in range(int(np.max(trace))//step):
    new_binary_trace = np.zeros((len(trace)))
    new_binary_trace = np.where(trace< old_threshold, new_binary_trace, np.max(trace))
    
    diff_binary_trace = np.diff(new_binary_trace)
    indexes = np.argwhere(np.diff(new_binary_trace)).squeeze()
    print( "\n", old_threshold)
    print(indexes.size, np.sum(diff_binary_trace))
    sum_off_trace = np.mean(trace[np.where(trace < old_threshold)])
    print(sum_off_trace)
#    plt.plot(new_binary_trace); plt.plot(trace); plt.plot(np.ones(len(trace))*old_threshold)
#    plt.plot(np.ones(len(trace))*np.mean(sum_off_trace))
    distance.append(np.mean(np.diff(indexes)))
#    if indexes.size > 3:
#    if sum_off_trace < np.mean(trace[-20:]):
#        old_threshold +=3
#    elif sum_off_trace > 2*np.mean(trace[-20:]):
#        old_threshold -=3
#    else:
#        converge = True
#        print("CONVERGE at ", old_threshold)
#        break
#    i+=1
    old_threshold += step

the_i = np.where(np.array(distance)==np.max(np.nan_to_num(distance)))
old_threshold = 1 + step*int(np.min(the_i))
new_binary_trace = np.where(trace< old_threshold, new_binary_trace, np.max(trace))
plt.plot(new_binary_trace); plt.plot(trace); plt.plot(np.ones(len(trace))*old_threshold)




