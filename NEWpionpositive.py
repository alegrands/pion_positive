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
Em_cm = (s-(mn**2)+(mm)**2)/(2*w) #momentum energy(?)

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
ax1.set_xlabel(r'$p \perp ^2$ cut (GeV^2)', size=30)
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

PDF = PdfPages('NEWallcutspionpositive.pdf')

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

av1s = np.average(s1[:4])
av2s = np.average(s1[4:8])
av3s = np.average(s1[8:12])
av4s = np.average(s1[12:16])
av5s = np.average(s1[16:20])
av6s = np.average(s1[20:24])
av7s = np.average(s1[24:28])
av8s = np.average(s1[28:32])
av9s = np.average(s1[32:36])
av10s = np.average(s1[36:40])
av11s = np.average(s1[40:44])
av12s = np.average(s1[44:48])
av13s = np.average(s1[48:52])
av14s = np.average(s1[52:56])
av15s = np.average(s1[56:60])
av16s = np.average(s1[60:64])
av17s = np.average(s1[64:68])
av18s = np.average(s1[68:72])
av19s = np.average(s1[72:76])
av20s = np.average(s1[76:80])
av21s = np.average(s1[80:84])
av22s = np.average(s1[84:88])
av23s = np.average(s1[88:92])
av24s = np.average(s1[92:96])
av25s = np.average(s1[96:100])
av26s = np.average(s1[100:104])
av27s = np.average(s1[104:108])
av28s = np.average(s1[108:112])
av29s = np.average(s1[112:116])
av30s = np.average(s1[116:120])
av31s = np.average(s1[120:124])
av32s = np.average(s1[124:128])
av33s = np.average(s1[128:132])
av34s = np.average(s1[132:134])
    
avs = np.array([av1s,av2s,av3s,av4s,av5s,av6s,av7s,av8s,av9s,av10s,av11s,
                av12s,av13s,av14s,av15s,av16s,av17s,av18s,av19s,av20s,av21s,
                av22s,av23s,av24s,av25s,av26s,av27s,av28s,av29s,av30s,
                av31s,av32s,av33s,av34s])

av1c = np.average(cospt1[:4])
av2c = np.average(cospt1[4:8])
av3c = np.average(cospt1[8:12])
av4c = np.average(cospt1[12:16])
av5c = np.average(cospt1[16:20])
av6c = np.average(cospt1[20:24])
av7c = np.average(cospt1[24:28])
av8c = np.average(cospt1[28:32])
av9c = np.average(cospt1[32:36])
av10c = np.average(cospt1[36:40])
av11c = np.average(cospt1[40:44])
av12c = np.average(cospt1[44:48])
av13c = np.average(cospt1[48:52])
av14c = np.average(cospt1[52:56])
av15c = np.average(cospt1[56:60])
av16c = np.average(cospt1[60:64])
av17c = np.average(cospt1[64:68])
av18c = np.average(cospt1[68:72])
av19c = np.average(cospt1[72:76])
av20c = np.average(cospt1[76:80])
av21c = np.average(cospt1[80:84])
av22c = np.average(cospt1[84:88])
av23c = np.average(cospt1[88:92])
av24c = np.average(cospt1[92:96])
av25c = np.average(cospt1[96:100])
av26c = np.average(cospt1[100:104])
av27c = np.average(cospt1[104:108])
av28c = np.average(cospt1[108:112])
av29c = np.average(cospt1[112:116])
av30c = np.average(cospt1[116:120])
av31c = np.average(cospt1[120:124])
av32c = np.average(cospt1[124:128])
av33c = np.average(cospt1[128:132])
av34c = np.average(cospt1[132:134])
    
avc = np.array([av1c,av2c,av3c,av4c,av5c,av6c,av7c,av8c,av9c,av10c,av11c,
                av12c,av13c,av14c,av15c,av16c,av17c,av18c,av19c,av20c,av21c,
                av22c,av23c,av24c,av25c,av26c,av27c,av28c,av29c,av30c,
                av31c,av32c,av33c,av34c])

av1p = np.average(pt21[:4])
av2p = np.average(pt21[4:8])
av3p = np.average(pt21[8:12])
av4p = np.average(pt21[12:16])
av5p = np.average(pt21[16:20])
av6p = np.average(pt21[20:24])
av7p = np.average(pt21[24:28])
av8p = np.average(pt21[28:32])
av9p = np.average(pt21[32:36])
av10p = np.average(pt21[36:40])
av11p = np.average(pt21[40:44])
av12p = np.average(pt21[44:48])
av13p = np.average(pt21[48:52])
av14p = np.average(pt21[52:56])
av15p = np.average(pt21[56:60])
av16p = np.average(pt21[60:64])
av17p = np.average(pt21[64:68])
av18p = np.average(pt21[68:72])
av19p = np.average(pt21[72:76])
av20p = np.average(pt21[76:80])
av21p = np.average(pt21[80:84])
av22p = np.average(pt21[84:88])
av23p = np.average(pt21[88:92])
av24p = np.average(pt21[92:96])
av25p = np.average(pt21[96:100])
av26p = np.average(pt21[100:104])
av27p = np.average(pt21[104:108])
av28p = np.average(pt21[108:112])
av29p = np.average(pt21[112:116])
av30p = np.average(pt21[116:120])
av31p = np.average(pt21[120:124])
av32p = np.average(pt21[124:128])
av33p = np.average(pt21[128:132])
av34p = np.average(pt21[132:134])
    
avp = np.array([av1p,av2p,av3p,av4p,av5p,av6p,av7p,av8p,av9p,av10p,av11p,
                av12p,av13p,av14p,av15p,av16p,av17p,av18p,av19p,av20p,av21p,
                av22p,av23p,av24p,av25p,av26p,av27p,av28p,av29p,av30p,
                av31p,av32p,av33p,av34p])

av1d = np.average(dsigdt1[:4])
av2d = np.average(dsigdt1[4:8])
av3d = np.average(dsigdt1[8:12])
av4d = np.average(dsigdt1[12:16])
av5d = np.average(dsigdt1[16:20])
av6d = np.average(dsigdt1[20:24])
av7d = np.average(dsigdt1[24:28])
av8d = np.average(dsigdt1[28:32])
av9d = np.average(dsigdt1[32:36])
av10d = np.average(dsigdt1[36:40])
av11d = np.average(dsigdt1[40:44])
av12d = np.average(dsigdt1[44:48])
av13d = np.average(dsigdt1[48:52])
av14d = np.average(dsigdt1[52:56])
av15d = np.average(dsigdt1[56:60])
av16d = np.average(dsigdt1[60:64])
av17d = np.average(dsigdt1[64:68])
av18d = np.average(dsigdt1[68:72])
av19d = np.average(dsigdt1[72:76])
av20d = np.average(dsigdt1[76:80])
av21d = np.average(dsigdt1[80:84])
av22d = np.average(dsigdt1[84:88])
av23d = np.average(dsigdt1[88:92])
av24d = np.average(dsigdt1[92:96])
av25d = np.average(dsigdt1[96:100])
av26d = np.average(dsigdt1[100:104])
av27d = np.average(dsigdt1[104:108])
av28d = np.average(dsigdt1[108:112])
av29d = np.average(dsigdt1[112:116])
av30d = np.average(dsigdt1[116:120])
av31d = np.average(dsigdt1[120:124])
av32d = np.average(dsigdt1[124:128])
av33d = np.average(dsigdt1[128:132])
av34d = np.average(dsigdt1[132:134])
    
avd = np.array([av1d,av2d,av3d,av4d,av5d,av6d,av7d,av8d,av9d,av10d,av11d,
                av12d,av13d,av14d,av15d,av16d,av17d,av18d,av19d,av20d,av21d,
                av22d,av23d,av24d,av25d,av26d,av27d,av28d,av29d,av30d,
                av31d,av32d,av33d,av34d])

av1g = np.average(sig1[:4])
av2g = np.average(sig1[4:8])
av3g = np.average(sig1[8:12])
av4g = np.average(sig1[12:16])
av5g = np.average(sig1[16:20])
av6g = np.average(sig1[20:24])
av7g = np.average(sig1[24:28])
av8g = np.average(sig1[28:32])
av9g = np.average(sig1[32:36])
av10g = np.average(sig1[36:40])
av11g = np.average(sig1[40:44])
av12g = np.average(sig1[44:48])
av13g = np.average(sig1[48:52])
av14g = np.average(sig1[52:56])
av15g = np.average(sig1[56:60])
av16g = np.average(sig1[60:64])
av17g = np.average(sig1[64:68])
av18g = np.average(sig1[68:72])
av19g = np.average(sig1[72:76])
av20g = np.average(sig1[76:80])
av21g = np.average(sig1[80:84])
av22g = np.average(sig1[84:88])
av23g = np.average(sig1[88:92])
av24g = np.average(sig1[92:96])
av25g = np.average(sig1[96:100])
av26g = np.average(sig1[100:104])
av27g = np.average(sig1[104:108])
av28g = np.average(sig1[108:112])
av29g = np.average(sig1[112:116])
av30g = np.average(sig1[116:120])
av31g = np.average(sig1[120:124])
av32g = np.average(sig1[124:128])
av33g = np.average(sig1[128:132])
av34g = np.average(sig1[132:134])
    
avg = np.array([av1g,av2g,av3g,av4g,av5g,av6g,av7g,av8g,av9g,av10g,av11g,
                av12g,av13g,av14g,av15g,av16g,av17g,av18g,av19g,av20g,av21g,
                av22g,av23g,av24g,av25g,av26g,av27g,av28g,av29g,av30g,
                av31g,av32g,av33g,av34g])

#%% averaging points

cosines = np.array([cospt1[1], cospt1[2], cospt1[3], cospt1[4]])

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

    plt.title(r'$\gamma p \rightarrow \pi^+ n$', size=35)
    plt.ylabel(r'$\frac{d\sigma}{dt}$ $[\mu$bGeV$^{-2}]$', size =30)
    plt.yscale('log')
    plt.xlabel('s [$GeV^2$]', size = 30)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=30, width=2.5, length=10)

#%%

#excercise

avs1 = np.array([np.average(ss[:2]), np.average(ss[2:4]), np.average(ss[4:6]), np.average(ss[6:8]), np.average(ss[8:11])])
avd1 = np.array([np.average(dsigdts[:2]), np.average(dsigdts[2:4]), np.average(dsigdts[4:6]), np.average(dsigdts[6:8]), np.average(dsigdts[8:11])])
avg1 = np.array([np.average(sigs[:2]), np.average(sigs[2:4]), np.average(sigs[4:6]), np.average(sigs[6:8]), np.average(sigs[8:11])])
avc1 = np.array([np.average(coss[:2]), np.average(coss[2:4]), np.average(coss[4:6]), np.average(coss[6:8]), np.average(coss[8:11])])

def fit(x, A, C, N):
    return (A+C*i)*x**(-7)

plt.errorbar(avs1, avd1, avg1, fmt='o', marker='v', color='red')
popt, pcov = curve_fit(fit, avs1, avd1, sigma=avg1, maxfev= 5000)
#plt.semilogy(ss, fit((ss), *popt), color = 'g', linestyle = '--')

plt.title(r'$\gamma p \rightarrow \pi^+ n$', size=35)
plt.ylabel(r'$\frac{d\sigma}{dt}$ $[\mu$bGeV$^{-2}]$', size =30)
plt.yscale('log')
plt.xlabel('s [$GeV^2$]', size = 30)
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=30, width=2.5, length=10)

#%%

y_pred = expanded_fit((coss, ss), popt[0], popt[1], popt[2])
avyp = np.array([np.average(y_pred[:2]), np.average(y_pred[2:4]), np.average(y_pred[4:6]), np.average(y_pred[6:8]), np.average(y_pred[8:11])])
chi_squared = np.sum(((avd1-avyp)/avg1)**2)
redchi1 = (chi_squared)/(len(coss)-len(popt))
redchi0 = np.append(redchi0, redchi)
N = np.abs(popt[2])
N_err = pcov[2,2]**0.5
Nres0 = np.append(Nres0, N)
Nerr0 = np.append(Nerr0, N_err)



#getting the data
F = B.get_file('newdataHEP.py')
Eg_MeV = B.get_data(F, 'Eg') #energy of photon in MeV
w = B.get_data(F, 'w') #Energy at cm
t = B.get_data(F, 't')
dsigdt = B.get_data(F, 'dsigdt')
error = B.get_data(F, 'error')
s = w**2

#converting energy of photon Mev -> GeV
Eg = Eg_MeV/1000

#masses in Gev
mb = 0.938272081 #proton mass which is a byron
mm = 0.13957039 #pion- mass which is a meson
mn = 0.93956542 #mass of a neutron which is a byron

#Energies in GeV
Eg_cm = (s-mn**2)/(2*w)
Eb_cm = (s+(mb**2)-(mm**2))/(2*w)
Em_cm = (s-(mb**2)+(mm)**2)/(2*w) #meson energy

#momentum
Pg_cm = Eg_cm #momentum of gamma k=E1
Pb_cm = np.sqrt((Eb_cm)**2 - (mb)**2) 
Pm_cm = np.sqrt((Em_cm)**2 - (mm)**2)

cos = ((Eg_cm*Em_cm)+((t-(mm)**2)/2))/(Pg_cm*Pm_cm)

t1 = (2.*Pg_cm*Pm_cm*cos+mm**2-2.*Eg_cm*Em_cm)
#dsigdt = dsig_dcos/ (2.*Pg_cm*Pm_cm)

#calculating transverse momentum (pt2) in four parts
pt2_1 = (s-mn**2)**2/(4*s)
pt2_2 = (((s+mm**2-mb**2)**2)/(4*s)) - mm**2
pt2_3 = ((1/(4*s)*(s-mn**2)*(s+mm**2-mb**2))+(t-mm**2)/2)**2
pt2_4 = ((s-mn**2)**2)/(4*s)
pt2 = (((pt2_1)*(pt2_2)-(pt2_3))/pt2_4)

#cosine close to zero

mx = 0.15
mn = -0.15

#exclusion of cosine values
cospt1 = cos[(cos <= mx)&(cos >= mn)] #cosine of angle 85 to 90 but in rad
s1 = s[(cos <= mx)&(cos >= mn)]
dsigdt1 = dsigdt[(cos <= mx)&(cos >= mn)]
t1 = t[(cos <= mx)&(cos >= mn)]
pt21 = pt2[(cos <= mx)&(cos >= mn)]
error1 = error[(cos <= mx)&(cos >= mn)]

#array for making cuts 
alpha = np.arange(0.15*max(pt21), 0.8*max(pt21), max(pt21)/100) #max(pt21) is the max value of the transverse momentum
second = np.arange(0.8*max(pt21), 0.99*max(pt21), (max(pt21)-0.8*max(pt21))/6)
alpha = np.append(alpha, second)

def fit(x, A, C, N):
    return (A+C*i)*x**(-N)

#plt.figure(figsize=(18,9))
#popt, pcov = curve_fit(expanded_fit, (cospt1, s1), dsigdt1, sigma=error1 , maxfev=5000)
#plt.semilogy(s1, expanded_fit((s1), *popt), color = 'g', linestyle = '--')
plt.errorbar(s1, dsigdt1, yerr=error1, fmt='o', marker='v', color='violet', label = 'cos = -0.1')
plt.yscale('log')
#plt.ylabel('$\frac{d\sigma}{dt}$', size=30)
plt.xlabel('s')


ss1=np.append(ss,s)

dsig1 = np.append(avd1,dsigdt1)
sigma11 = np.append(avg1,error1)
cos11 = np.append(avc1,cospt1)
cos22 = np.append(coss, cospt1)
s11 = np.append(avs1,s1)
popt, pcov = curve_fit(fit, s11, dsig1, sigma=sigma11, maxfev= 5000)
plt.semilogy(ss1, fit((ss1), *popt), color = 'g', linestyle = '--')
#plt.semilogy(ss, fit((ss), *popt), color = 'g', linestyle = '--')

