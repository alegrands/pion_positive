#pion positive 
#gamma+p --> pi+ + n

import numpy as np
import LT.box as B
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

#getting the data
F = B.get_file('pionpositive.data.py')
Eg_MeV = B.get_data(F, 'Egamma') #energy of photon in MeV
cos = B.get_data(F, 'cos(theta)') #cosine of angle dimensionless
dsig_domega = B.get_data(F, 'dsigma/dOmega') #differential cross section in mcb/sr
d_dsigdomega = B.get_data(F, 'Data errors') #data errors in mcb/sr

#converting energy of photon Mev -> GeV
Eg = Eg_MeV/1000

#masses in Gev
mb = 0.938272081 #proton mass which is a byron
mm = 0.13957039 #pion- mass which is a meson
mn = 0.93956542 #mass of a neutron which is a byron

#getting s by using 4-vector
s = (mb)**2+(2*Eg*mb)
w = np.sqrt(s) #total energy of particles colliding

#converting dsig/domega to dsig/cos
dsig_dcos = 2*np.pi*dsig_domega
d_dsigdcos = 2*np.pi*dsig_domega

#Energies in GeV
Eg_cm = (s-mb**2)/(2*w)
Eb_cm = (s+(mn**2)-(mm**2))/(2*w)
Em_cm = (s-(mn**2)+(mm)**2)/(2*w) 

#momentum
Pg_cm = Eg_cm #momentum of gamma k=E1
Pb_cm = np.sqrt((Eb_cm)**2 - (mb)**2) 
Pm_cm = np.sqrt((Em_cm)**2 - (mm)**2)

t = (2.*Pg_cm*Pm_cm*cos+mm**2-2.*Eg_cm*Em_cm)
dsigdt = dsig_dcos/ (2.*Pg_cm*Pm_cm)
d_dsigdt = d_dsigdcos/(2.*Pg_cm*Pm_cm) #ask abt this

#calculating transverse momentum (pt2) in four parts
pt2_1 = (s-mb**2)**2/(4*s)
pt2_2 = (((s+mm**2-mn**2)**2)/(4*s)) - mm**2
pt2_3 = ((1/(4*s)*(s-mb**2)*(s+mm**2-mn**2))+(t-mm**2)/2)**2
pt2_4 = ((s-mb**2)**2)/(4*s)
pt2 = (((pt2_1)*(pt2_2)-(pt2_3))/pt2_4)

#cosine close to zero
mx = 0.15
mn = -0.15

#exclusion of cosine values
cospt1 = cos[(cos <= mx)&(cos >= mn)] #cosine of angle 85 to 90 but in rad
s1 = s[(cos <= mx)&(cos >= mn)]
dsigdt1 = dsigdt[(cos <= mx)&(cos >= mn)]
sig1 = dsig_domega[(cos <= mx)&(cos >= mn)] #is it necessary?
t1 = t[(cos <= mx)&(cos >= mn)]
pt21 = pt2[(cos <= mx)&(cos >= mn)]
d_dsigdt1 = d_dsigdt[(cos <= mx)&(cos >= mn)]


#array for making cuts 
alpha = np.arange(0.15*max(pt21), 0.8*max(pt21), max(pt21)/100) #max(pt21) is the max value of the transverse momentum
second = np.arange(0.8*max(pt21), 0.99*max(pt21), (max(pt21)-0.8*max(pt21))/6)
alpha = np.append(alpha, second)

def expanded_fit(x, A, C, N):
    return (A + C*x[0])*x[1]**(-N)

plt.figure(figsize=(18,9))
popt, pcov = curve_fit(expanded_fit, (cospt1, s1), dsigdt1, sigma=sig1, maxfev=5000)
plt.errorbar(s1, dsigdt1, yerr=sig1, fmt='o', marker='v', color='g')
plt.yscale('log')
#plt.ylabel('$\frac{d\sigma}{dt}$', size=30)
plt.xlabel('s')

redchi0 = np.array([])
Nres0 = np.array([])
Nerr0 = np.array([])
cut = np.array([])
perc = np.array([])

for j in alpha[:-2]:
    
    per = j/max(alpha)
    perc = np.append(perc, per)
    cut = np.append(cut, j)
    pt2min = j
    
    #exclude values lower than the min transverse momentum
    coss = cospt1[pt21 >= pt2min]
    dsigdts = dsigdt1[pt21 >= pt2min]
    sigs = sig1[pt21 >= pt2min]
    pts = pt21[pt21 >= pt2min]
    ts = -t1[pt21 >= pt2min]
    ss = s1[pt21 >= pt2min]
    
    #fit
    popt, pcov = curve_fit(expanded_fit, (coss, ss), dsigdts, sigma=sigs, maxfev=5000)
    plt.errorbar(ss, dsigdts, yerr=sigs, fmt='o', marker='v', color='g')
    plt.yscale('log')
    #plt.title(r'$\frac{d\sigma}{dt} = (A+Bcos\theta)s^-N; \gamma  n \rightarrow \pi^- p$', size = 30)
    #plt.ylabel(r'$\frac{d\sigma}{dt}$', size=30)
    #plt.xlabel(r'$s (GeV^2)$', size=30)
    plt.show()
    
    y_pred = expanded_fit((coss, ss), popt[0], popt[1], popt[2])
    chi_squared = np.sum(((dsigdts-y_pred)/sigs)**2)
    redchi = (chi_squared)/(len(coss)-len(popt))
    redchi0 = np.append(redchi0, redchi)
    N = np.abs(popt[2])
    N_err = pcov[2,2]**0.5
    Nres0 = np.append(Nres0, N)
    Nerr0 = np.append(Nerr0, N_err)
#%%
plt.figure(figsize=(18,10))
fig, ax1 = plt.subplots()
plt.title(r'$\gamma p \rightarrow \pi^+ n$', size=30)
color = 'tab:red'
ax1.set_xlabel(r'$p_T ^2$ cut (GeV^2)', size=30)
ax1.set_ylabel(r'$\chi^2/df$', size=30, color='tab:red')
ax1.plot(cut, redchi0, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
ax2 = ax1.twinx() #instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r"$N$", size=30, color='tab:blue')
ax2.errorbar(cut, Nres0, Nerr0, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.rc("xtick", labelsize=25)
plt.rc("ytick", labelsize=30) 
plt.show() 

#%%

#loop to have the cosines in the same graph

cosines = np.array([cospt1[1], cospt1[2], cospt1[3], cospt1[4]]) #-0.15,-0.05,0.05,0.15

def fit(x, A, C, N):
    return (A + C*i)*x**(-N)

for i in cosines:
    plt.figure(figsize=(18,9))
    if (i == cospt1[1]):
        coss = cospt1[(cospt1 == i)&(pt21>j)]
        dsigdts = dsigdt1[(cospt1 == i)&(pt21>j)]
        sigs = sig1[(cospt1 == i)&(pt21>j)]
        ss = s1[(cospt1 == i)&(pt21>j)]
        pts = pt21[(cospt1 == i)&(pt21>j)]
        plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'g')
        popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
    elif (i == cospt1[2]):
        coss = cospt1[(cospt1 == i)&(pt21>j)]
        dsigdts = 2*dsigdt1[(cospt1 == i)&(pt21>j)]
        sigs = 2*sig1[(cospt1 == i)&(pt21>j)]
        ss = s1[(cospt1 == i)&(pt21>j)]
        pts = pt21[(cospt1 == i)&(pt21>j)]
        plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'g')
        popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
    elif (i == cospt1[3]):
        coss = cospt1[(cospt1 == i)&(pt21>j)]
        dsigdts = 4*dsigdt1[(cospt1 == i)&(pt21>j)]
        sigs = 4*sig1[(cospt1 == i)&(pt21>j)]
        ss = s1[(cospt1 == i)&(pt21>j)]
        pts = pt21[(cospt1 == i)&(pt21>j)]
        plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'g')
        popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
    elif (i == cospt1[4]):
        coss = cospt1[(cospt1 == i)&(pt21>j)]
        dsigdts = 8*dsigdt1[(cospt1 == i)&(pt21>j)]
        sigs = 8*sig1[(cospt1 == i)&(pt21>j)]
        ss = s1[(cospt1 == i)&(pt21>j)]
        pts = pt21[(cospt1 == i)&(pt21>j)]
        plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'g')
        popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
        
    plt.title(r'$\gamma p \rightarrow \pi^+ n$', size=35)
    plt.ylabel(r'$\frac{d\sigma}{dt}$ $[\mu$bGeV$^{-2}]$', size =30)
    plt.yscale('log')
    plt.xlabel('s [$GeV^2$]', size = 30)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=30, width=2.5, length=10)
    
#%%
#another loop to cut the data

PDF = PdfPages('allcutspionpositive.pdf')

for j in alpha[:-5]:
    
    def fit(x, A, C, N):
        return (A+C*i)*x**(-N)

    plt.figure(figsize=(18,9))
    for i in cosines:
        if (i == cospt1[1]):
            coss= cospt1[(cospt1 == i)&(pt21>j)]
            dsigdts = dsigdt1[(cospt1 == i)&(pt21>j)]
            sigs = sig1[(cospt1 == i)&(pt21>j)]
            ss = s1[(cospt1 == i)&(pt21>j)]
            pts = pt21[(cospt1 == i)&(pt21>j)]
            plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'g')
            popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
            plt.semilogy(ss, fit((ss), *popt), color = 'g', linestyle = '--')
        elif (i == cospt1[2]):
             coss= cospt1[(cospt1 == i)&(pt21>j)]
             dsigdts = 2*dsigdt1[(cospt1 == i)&(pt21>j)]
             sigs = 2*sig1[(cospt1 == i)&(pt21>j)]
             ss = s1[(cospt1 == i)&(pt21>j)]
             pts = pt21[(cospt1 == i)&(pt21>j)]
             plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'b')
             popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
             plt.semilogy(ss, fit((ss), *popt), color = 'b', linestyle = '--')
        elif (i == cospt1[3]):
             coss= cospt1[(cospt1 == i)&(pt21>j)]
             dsigdts = 4*dsigdt1[(cospt1 == i)&(pt21>j)]
             sigs = 4*sig1[[(cospt1 == i)&(pt21>j)]]
             ss = s1[(cospt1 == i)&(pt21>j)]
             pts = pt21[[(cospt1 == i)&(pt21>j)]]
             plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'r')
             popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
             plt.semilogy(ss, fit((ss), *popt), color = 'r', linestyle = '--')
        elif (i == cospt1[4]):
             coss= cospt1[(cospt1 == i)&(pt21>j)]
             dsigdts = 8*dsigdt1[(cospt1 == i)&(pt21>j)]
             sigs = 8*sig1[(cospt1 == i)&(pt21>j)]
             ss = s1[(cospt1 == i)&(pt21>j)]
             pts = pt21[(cospt1 == i)&(pt21>j)]
             plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'orange')
             popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
             plt.semilogy(ss, fit((ss), *popt), color = 'orange', linestyle = '--')
             plt.legend(['$\cos \Theta = -0.15$','$\cos \Theta = -0.05$','$\cos \Theta = 0.05$' ,'$\cos \Theta = 0.15$' ],loc = 'lower left', fontsize = 19)
        
        #format   
        #plt.title(r'$\gamma  p \rightarrow \omega p$: $\frac{d\sigma}{dt}$=$(A + B \cos \Theta)s^{-6.60 \pm 0.04}$, $p_{\perp _{min}} ^2 = 0.003}$, $\chi ^2 /df = 78$'  , size = 35)
    plt.title(r'$\gamma p \rightarrow \pi^+ n$', size=35)
    plt.ylabel(r'$\frac{d\sigma}{dt}$ $[\mu$bGeV$^{-2}]$', size =30)
    plt.yscale('log')
    plt.xlabel('s [$GeV^2$]', size = 30)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=30, width=2.5, length=10)
    PDF.savefig()
    
PDF.close()

#%%


     

        
        

        
    








