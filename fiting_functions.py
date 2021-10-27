# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:32:39 2021

@author: chiarelg
"""

def plot_histo_fit(vector, bines, name=" ", shift=0.5, double=False, printing=False, ploting=False):
    import numpy as np
    import matplotlib.pyplot as plt
    
#    import pylab as plb
    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp

    N = len(vector)/bines
    bins = np.linspace(0,int(np.max(vector)), N)
    if ploting==True:
        plt.hist(vector, bins=bins, alpha = 0.5, label=name)# , color="#900090",alpha=0.6,label='data')  # len(nozeros)//N
    y,x = np.histogram(vector, bins=bins)  #len(nozeros)//N
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    
#    n = len(x)                          #the number of data
    mean = sum(x*y)/sum(y)                   #note this correction
    sigma = sum(y*(x-mean)**2)/sum(y)        #note this correction
    
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    try:
        popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma/100])
        perr = np.sqrt(np.diag(pcov))
               
        
        if ploting==True:
            #plt.plot(x,y,'b+:',label='data')
            X = np.linspace(x[0], x[-1], 500)
            plt.plot()
            plt.plot(X,gaus(X,*popt),'g',lw=2, label='1G fit')
            plt.vlines(popt[1], color="k", ymin=0,ymax=0.5*popt[0])
            plt.vlines((popt[1]-popt[2], popt[1]+popt[2]),color='orange', ymin=0, ymax=10)
            plt.legend()
            plt.title('hist')
            plt.xlabel('Photons')
            plt.ylabel("total points ={} in {} bins".format(len(vector), N))
    #    plt.xlim([0,3000])
        #    plt.text(30,50, "mean ={:.2f}±{:.2f}".format(popt[1], popt[2]))
        if printing == True:
            print(popt[1], popt[2], "Gauss 1D")
        #plt.xlim(np.min(x), popt[1]+abs(popt[2]*3))
        #plt.xlim(0, int(np.max(nozeros)))
        #plt.show()
    except:
        print("bad adjusts 1G !!!")
        return (0,0)
        
    
    def gauss(x,mu,sigma,A):
        return A*exp(-(x-mu)**2/2/sigma**2)
    
    def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
        return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)


    if double == False:
        return popt[1], popt[2]
    else:
        try:
            try:
                expected = (popt[1],abs(popt[2]),popt[0],
                        shift*popt[1], 0.5*abs(popt[2]), 0.5*popt[0])
            except:
                expected = (1,mean,sigma/100, shift*mean,0.5*(abs(sigma/100)), 0.5)
            
            params,cov = curve_fit(bimodal,x,y,expected)
            sigma = np.sqrt(np.diag(cov))
        
            if params[0] < 0 or params[3] < 0:
                print("bad adjusts 2G")
            else:
                if ploting==True:
                    #X = np.linspace(x[0]-50, x[-1]+50, 5000)
                    plt.plot(X,bimodal(X,*params),color='orange',lw=3,label='2G model')
                    plt.legend()
                    plt.vlines((params[0], params[3]), color=('r','b'), ymin=0,ymax=0.5*popt[0])
                
        #        print(params,'\n',sigma)
                #print("\n mal Gauss", (viejopopt[1],"±", viejopopt[2]),"*",viejopopt[0])
                if printing == True:
                    print(params[0], params[3], "Gauss 2D")
            
            return (int(params[0]), int(params[3]))
        except:
            params = ["no"]*6
            return popt[1], popt[1]
    
#    print("\n 1Gaus=",(popt[1],"±", popt[2]), "*", popt[0])
#    print("\n 2Gaus=",(params[0], "±", params[1]), "*", params[2],
#              "\n",(params[3],"±",params[4]), "*", params[5])


