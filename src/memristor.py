from _context import *
# sys.path.append("../../optolab_control_software") 
# from meas import *
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cmath
from scipy import optimize
import random
import warnings
ch_DAQ_AWG, ch_DAQ_SMU  = 113, 111

class Memristor:
    # -------------------------------------------------
        # Memristor Model Parameters
    # -------------------------------------------------
    #ID of Memristor
    ID=""
    #Number of Pulses to max Conductance
    N=200

    #Normalization
    G0=1/1e4
    G1=1/1e3

    #--Model Parameters---
    #One Model
    alpha_set=-0.005
    beta_set=0.05 
    alpha_reset=-0.005
    beta_reset=0.05 
    gamma_set=1
    gamma_reset=1
    zeta_set=0
    zeta_reset=0

    #Model Parameters per Cycle
    alpha_noise_set=np.array([]) 
    beta_noise_set=np.array([]) 
    gamma_noise_set=np.array([]) 
    zeta_noise_set=np.array([]) 
    alpha_noise_reset=np.array([]) 
    beta_noise_reset=np.array([]) 
    gamma_noise_reset=np.array([]) 
    zeta_noise_reset=np.array([])


    #--Noise parameters--
    #Constant
    std=0.2/6
    #Separate for Set,Reset (1 Parameter for all cycles)
    std_noise_set=0.05 
    std_noise_reset=0.05
    #Per Cycle
    std_noise_set_cycle=np.array([]) 
    std_noise_reset_cycle=np.array([]) 
    #For choosing cycle
    sign_counter=0 #Counts from 0 to 2. If at 2 and sign change occurs, counter is reset 
    update_sign=None #sign of last update: +1 for positive update, -1 for negative 
    cycle=None #Current cycle
    count_direction='up' #direction in which cycle is selected 

    
    #Data 
    Gset=np.array([]) 
    Greset=np.array([]) 
    pset=np.array([]) 
    preset=np.array([]) 
    Gset_norm=np.array([]) 
    Greset_norm=np.array([]) 
     
    # -------------------------------------------------
        # Waveforms and Measurment Setting Read and Write
    # -------------------------------------------------
    wf_write = {
            'V' : [1,-1], # pulse voltages
            'n' : [100,100],   # pulse repetitions
            'W' : [10e-6,10e-6],     # pulse widths
            'T' : 500e-6,     # pulse period
            'read_offset' : True,
            'read_V' : 0.1,
            'spp':10,
            'srate':None,
            'output_impedance':1e6,
            # 'chs_awg':[1,2]
        }
    
    wf_read = {     # Read Waveform 
            'V' : [0], # pulse voltages
            'n' : [1],   # pulse repetitions
            'W' : [1e-3],     # pulse widths
            'T' : 21e-3,     # pulse period
            'read_offset' : True,
            'read_V' : 0.1,
            'output_impedance':1e6,
            'srate':1e6,
            'spp':None
        }
    #Measurement settings
    meas_dict_write={
            'min_risetime':(wf_write['T']-wf_write['W'][0])/10,
            'R_s':1e3,
            'R_min': 1e4,
            'gain': None, 
            'V_scale':[wf_write['read_V'],wf_write['read_V']],
            'input_impedance': [1e6,1e6],
            'ref_src': 'AWG', #reference output src
            'ref_ch':2, #channel of reference output
            'device_src': 'AWG', #src where device is connected to
            'device_ch': 1, #output channel that is connected to device
        }
    
    meas_dict_read={
            'min_risetime':(wf_read['T']-wf_read['W'][0])/10, 
            'R_s':1e3,
            'R_min': 1e4,
            'gain': None, 
            'V_scale':[wf_read['read_V'],wf_read['read_V']],
            'input_impedance': [1e6,1e6],
            'ref_src': 'AWG', #reference output src
            'ref_ch':2, #channel of reference output
            'device_src': 'AWG', #src where device is connected to
            'device_ch': 1, #output channel that is connected to device
        }
    
    #Parameters to be overwritten when amplifier used:
    amplifier_settings={
        'R_s':0,
        'gain': 1e4,
        'gain_delta_w': 1e5,
        'low_high_gain':'low',
        'amp_type': 'DHPCA_100',
        'input_impedance': [1e6,50],
    }
    
    def __init__(self, params=None,old_version=False):
       #Load Parameters
        if params is not None:
            if old_version:
                self.params = params
                self.N=params['N'] #number of pulses to max conductance
                self.alpha_set=params['popt']['s'][0] #parameter nonlinear update fct
                self.alpha_reset=params['popt']['r'][0] #parameter nonlinear update fct
                self.beta_set=params['popt']['s'][1] #parameter nonlinear update fct
                self.beta_reset=params['popt']['r'][1] #parameter nonlinear update fct
            else:
                self.update_params(params)
    
    def update_params(self, params):
        self.N=params['N'] #number of pulses to max conductance
        self.beta_set=params['beta_set'] #parameter nonlinear update fct
        self.beta_reset=params['beta_reset'] #parameter nonlinear update fct
        self.G0=params['G0']
        self.G1=params['G1']
        if 'alpha_set' in params.keys():
            self.alpha_set=params['alpha_set']
            self.alpha_reset=params['alpha_reset']
        if 'alpha_noise_set' in params.keys():
            self.alpha_noise_set=params['alpha_noise_set']
            self.beta_noise_set=params['beta_noise_set']
            self.gamma_noise_set=params['gamma_noise_set']
            self.zeta_noise_set=params['zeta_noise_set']
            self.alpha_noise_reset=params['alpha_noise_reset']
            self.beta_noise_reset=params['beta_noise_reset']
            self.gamma_noise_reset=params['gamma_noise_reset'] 
            self.zeta_noise_reset=params['zeta_noise_reset']
        if 'std_noise_set' in params.keys():
            self.std_noise_set=params['std_noise_set']
            self.std_noise_reset=params['std_noise_reset']  
        if 'std_noise_set_cycle' in params.keys():
            self.std_noise_set_cycle=params['std_noise_set_cycle']
            self.std_noise_reset_cycle=params['std_noise_reset_cycle']
        if 'Gset' in params.keys():
            self.Gset=params['Gset']
            self.Greset=params['Greset']
            self.pset=params['pset']
            self.preset=params['preset']
            self.Gset_norm=params['Gset_norm']
            self.Greset_norm=params['Greset_norm']

    def reset_counter(self,n):
        self.sign_counter=0
        self.update_sign = None
        self.count_direction = 'up'
        self.cycle = None

    # -------------------------------------------------
        # Memristor Model Functions
    # -------------------------------------------------
    
    #-----------------------------------------------------------
    # Linear and Exponential added together

    def Lin_Exp_Set(self, p_,alpha=None,beta=None):
        if alpha is None:
            alpha=self.alpha_set
        if beta is None:
            beta=self.beta_set
        p = np.clip(p_, 0, self.N)
        return (1 - (alpha*p +np.exp(-p * beta))) / (1 - (alpha*self.N + np.exp(-self.N * beta)))

    def Lin_Exp_Reset(self, p_,alpha=None,beta=None):
        if alpha is None:
            alpha=self.alpha_reset
        if beta is None:
            beta=self.beta_reset
        p = np.clip(p_, 0, self.N)
        return 1 - (1 - (alpha*(self.N - p) +np.exp(-(self.N - p) * beta))) / (1 - (alpha*self.N + np.exp(-self.N * beta)))
    
    #-----------------------------------------------------------
    # Linear and Exponential added together with 3 parameters
        
    #b0+b1*inp+b2*np.exp(alpha*inp)
    #(zeta - (alpha*p +zeta*np.exp(-p * beta))) /(zeta - (alpha*N +zeta*np.exp(-beta * N))) 
    
    def Lin_Exp_Set_3params(self, p_,alpha=None,beta=None,gamma=None):
        if alpha is None:
            alpha=self.alpha_set
        if beta is None:
            beta=self.beta_set
        if gamma is None:
            gamma=self.gamma_set
        p = np.clip(p_, 0, self.N)
        return (1 - (alpha*p +gamma*np.exp(-p * beta))) / (1 - (alpha*self.N + gamma*np.exp(-self.N * beta)))

    def Lin_Exp_Reset_3params(self, p_,alpha=None,beta=None, gamma=None):
            if alpha is None:
                alpha=self.alpha_reset
            if beta is None:
                beta=self.beta_reset
            if gamma is None:
                gamma=self.gamma_reset
            p = np.clip(p_, 0, self.N)
            return 1 - (1 - (alpha*(self.N-p)+gamma*np.exp(-(self.N-p)*beta)))/(1-(alpha*self.N+gamma*np.exp(-self.N*beta)))

    #-----------------------------------------------------------
    # Const + Linear + Exponential added together with 4 parameters

    def Lin_Exp_Set_4params(self, p_,alpha=None,beta=None, gamma=None, zeta=None,clip=False): #Not clipped!
        if type(p_) in [np.float64,float,int]:
            if (-self.N<p_<2*self.N)==False:
                print('warning: p is far away from safe p-range of function',p_)
        else:
            if all(-self.N<p_)==False or all(p_<2*self.N)==False:
                print('warning: p is far away from safe p-range of function')

        if alpha is None:
            alpha=self.alpha_set
        if beta is None:
            beta=self.beta_set
        if gamma is None:
            gamma=self.gamma_set
        if zeta is None:
            zeta=self.zeta_set

        if clip:
            p = np.clip(p_, 0, self.N)
        else:
            p=p_
        return alpha+(beta*p)-(gamma*np.exp(-zeta*p))
    
    def Lin_Exp_Reset_4params(self, p_,alpha=None,beta=None, gamma=None, zeta=None,clip=False): #Not clipped!
        if type(p_) in [np.float64,float,int]:
            if (-self.N<p_<2*self.N)==False:
                print('warning: p is far away from safe p-range of function',p_)
        else:
            if all(-self.N<p_)==False or all(p_<2*self.N)==False:
                print('warning: p is far away from safe p-range of function')

        if alpha is None:
            alpha=self.alpha_reset
        if beta is None:
            beta=self.beta_reset
        if gamma is None:
            gamma=self.gamma_reset
        if zeta is None:
            zeta=self.zeta_reset

        if clip:
            p = np.clip(p_, 0, self.N)
        else:
            p=p_
        return alpha+(beta*p)+(gamma*np.exp(+zeta*p))

    def Lin_Exp0_Set_4params(self, p_,alpha=None,beta=None, zeta=None):
        if alpha is None:
            alpha=self.alpha_set
        if beta is None:
            beta=self.beta_set
        if gamma is None:
            gamma=self.gamma_set
        if zeta is None:
            zeta=self.zeta_set
        p = np.clip(p_, 0, self.N)
        return (beta*p)+alpha*(1-np.exp(-zeta*p))
    
    def Lin_Exp0_Reset_4params(self, p_,alpha=None,beta=None, zeta=None):
        if alpha is None:
            alpha=self.alpha_reset
        if beta is None:
            beta=self.beta_reset
        if gamma is None:
            gamma=self.gamma_reset
        if zeta is None:
            zeta=self.zeta_reset
        p = np.clip(p_, 0, self.N)
        return (beta*p)+alpha*(1+np.exp(zeta*p))

    #-----------------------------------------
    #---Inverse Functions with Newton Method---


    #-----Inverse for Linexp with 4 Parameters------

    # SET:
    def Lin_Exp_Set_4params_zero_prime(self, x,alpha,beta,gamma,zeta): #First derivative of that function
        return -beta-zeta*gamma*np.exp(-zeta*x)
    
    def Lin_Exp_Set_4params_zero(self,x,y,alpha,beta, gamma, zeta,clip=False): #Function where 0 should be found
            return y-self.Lin_Exp_Set_4params(x,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta,clip=clip)
    
    def Lin_Exp_Set_4params_INV(self, G_, alpha=None, beta=None,gamma=None, zeta=None,clip=False):
        if alpha is None:
            alpha=self.alpha_set
        if beta is None:
            beta=self.beta_set
        if gamma is None:
            gamma=self.gamma_set
        if zeta is None:
            zeta=self.zeta_set
        
        if clip==True:
            G = np.clip(G_, 0, 1)
        elif clip=='Glim': # #Since function does not go to 0 and 1 necessarily
            Gmin=self.Lin_Exp_Set_4params(0,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta)
            Gmax=self.Lin_Exp_Set_4params(self.N,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta)
            G = np.clip(G_, Gmin, Gmax)
        else:
            G=G_

        #initial guess
        x0=self.Pset_beta(G,beta=0.015) #fixed zeta (here beta) that it does not become infinite close to G=1
        
        f_zero=lambda x : self.Lin_Exp_Set_4params_zero(x,y=G,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta,clip=False)
        f_zero_prime=lambda x : self.Lin_Exp_Set_4params_zero_prime(x,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta)
        result=optimize.newton(f_zero,x0=x0,fprime=f_zero_prime,tol=1e-3,maxiter=50) #Newton/Newton–Raphson method to find zero -> here to find pulse for given conductance
            
        return result
    
    # RESET:
    def Lin_Exp_Reset_4params_zero_prime(self, x,alpha,beta,gamma,zeta): #First derivative of that function
        return -beta-zeta*gamma*np.exp(zeta*x)

    def Lin_Exp_Reset_4params_zero(self,x,y,alpha,beta,gamma,zeta,clip=False): #Function where 0 should be found
            return y-self.Lin_Exp_Reset_4params(x,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta,clip=clip)
    
    def Lin_Exp_Reset_4params_INV(self, G_, alpha=None, beta=None,gamma=None, zeta=None,clip=False):
        if alpha is None:
            alpha=self.alpha_reset
        if beta is None:
            beta=self.beta_reset
        if gamma is None:
            gamma=self.gamma_reset
        if zeta is None:
            zeta=self.zeta_reset
        
        if clip==True:
            G = np.clip(G_, 0, 1)
        elif clip=='Glim': # #Since function does not go to 0 and 1 necessarily
            Gmin=self.Lin_Exp_Reset_4params(0,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta)
            Gmax=self.Lin_Exp_Reset_4params(self.N,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta)
            G = np.clip(G_, Gmin, Gmax)
        else:
            G=G_

        #initial guess
        x0=self.Preset_beta(G,beta=0.015) #fixed zeta (here beta) that it does not become infinite close to G=1

        f_zero=lambda x : self.Lin_Exp_Reset_4params_zero(x,y=G,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta,clip=False) #do not clip as this is only to find inverse function
        f_zero_prime=lambda x : self.Lin_Exp_Reset_4params_zero_prime(x,alpha=alpha,beta=beta,gamma=gamma,zeta=zeta)
        result=optimize.newton(f_zero,x0=x0,fprime=f_zero_prime,tol=1e-3,maxiter=50) #Newton/Newton–Raphson method to find zero -> here to find pulse for given conductance

        return result

    #-----Inverse for Linexp with 2 Parameters---
    def Lin_Exp_Set_zero_prime(self, x,alpha,beta): #First derivative of that function
        return (alpha -beta*np.exp(-x * beta)) / (1 - (alpha*self.N + np.exp(-self.N * beta)))
    
    def Lin_Exp_Set_zero_prime2(self, x,alpha,beta): #Second derivative of that function
        return (beta**2*np.exp(-x * beta)) / (1 - (alpha*self.N + np.exp(-self.N * beta)))
    
    def Lin_Exp_Set_zero(self,x,y,alpha,beta): #Function where 0 should be found
            return y-self.Lin_Exp_Set(x,alpha=alpha,beta=beta)
    
    def Lin_Exp_Set_INV(self, G_, alpha=None, beta=None):
        if alpha is None:
            alpha = self.alpha_set
        if beta is None:
            beta = self.beta_set
        G = np.clip(G_, 0, 1)

        #initial guess (use linear function btw. 0 and 1 in N steps)
        # a=1/self.N
        x0=self.Pset_beta(G,beta=0.015)
        f_zero=lambda x : self.Lin_Exp_Set_zero(x,y=G,beta=beta,alpha=alpha)
        f_zero_prime=lambda x : self.Lin_Exp_Set_zero_prime(x,beta=beta,alpha=alpha)
        f_zero_prime2=lambda x : self.Lin_Exp_Set_zero_prime2(x,beta=beta,alpha=alpha)

        result=optimize.newton(f_zero,fprime=f_zero_prime,x0=x0,tol=1e-3,maxiter=50) #Newton/Newton–Raphson method to find zero -> here to find pulse for given conductance
        return result
    
    def Lin_Exp_Reset_zero_prime(self, x,alpha,beta): #First derivative of that function
        return (alpha -beta*np.exp(-(self.N - x) * beta)) / (1 - (alpha*self.N + np.exp(-self.N * beta)))
    
    def Lin_Exp_Reset_zero_prime2(self, x,alpha,beta): #Second derivative of that function
        return (-beta**2*np.exp(-(self.N - x) * beta)) / (1 - (alpha*self.N + np.exp(-self.N * beta)))
    
    def Lin_Exp_Reset_zero(self,x,y,alpha,beta): #Function where 0 should be found
            return y-self.Lin_Exp_Reset(x,alpha=alpha,beta=beta)

    def Lin_Exp_Reset_INV(self, G_, alpha=None, beta=None):
        if alpha is None:
            alpha = self.alpha_reset
        if beta is None:
            beta = self.beta_reset
        G = np.clip(G_, 0, 1)

        #initial guess (use linear function btw. 0 and 1 in N steps)
        # a=1/self.N
        # x0=G/a
        x0=self.Preset_beta(G,beta=0.05)

        f_zero=lambda x : self.Lin_Exp_Reset_zero(x,y=G,beta=beta,alpha=alpha)
        f_zero_prime=lambda x : self.Lin_Exp_Reset_zero_prime(x,beta=beta,alpha=alpha)
        f_zero_prime2=lambda x : self.Lin_Exp_Reset_zero_prime2(x,beta=beta,alpha=alpha)
        return optimize.newton(f_zero,fprime=f_zero_prime,x0=x0,tol=1e-3,maxiter=50) #Newton/Newton–Raphson method to find zero -> here to find pulse for given conductance
   
    #Calc Number of Pulses between two conductances
    def n_pulses(self,G_i,G_f,clipped=True):
        """
        Get delta pulse number from current and new conductance values.
        """
        if clipped:
            G_i = np.clip(G_i, 0, 1)
            G_f = np.clip(G_f, 0, 1)
        if G_f > G_i:
            return self.Lin_Exp_Set_INV(G_f) - self.Lin_Exp_Set_INV(G_i)
        else:
            return self.Lin_Exp_Reset_INV(G_f) - self.Lin_Exp_Reset_INV(G_i)
        
    #Calc Number of Pulses between two conductances for linear update
    def n_pulses_lin(self,G_i,G_f,clipped=True):
        """
        Get delta pulse number from current and new conductance values.
        """
        if clipped:
            G_i = np.clip(G_i, 0, 1)
            G_f = np.clip(G_f, 0, 1)
        return self.N*(G_f) - self.N*(G_i)
    

    #----------------------------------------------
    # Nonlinear function with beta

    def Gset_beta(self, p_,beta=None): #attention -> defaults set at creation of class -> (self,G,beta=beta) not updated
        if beta is None:
            beta=self.beta_set
        p = np.clip(p_, 0, self.N)
        return (1 - np.exp(-p * beta)) / (1 - np.exp(-self.N * beta))

    def Greset_beta(self, p_,beta=None):
        if beta is None:
            beta=self.beta_reset
        p = np.clip(p_, 0, self.N)
        return 1 - (1 - np.exp(-(self.N - p) * beta)) / (1 - np.exp(-self.N * beta))
    
    #Inverse Functions
    def Pset_beta(self, G_,beta=None):
        if beta is None:
            beta=self.beta_set
        G = np.clip(G_, 0, 1)
        G_inf = 1/(1-np.exp(-beta*self.N))
        if type(G) in [np.float64,float,int]:
            assert (-np.log(1-G/G_inf)/beta) !=np.inf and (-np.log(1-G/G_inf)/beta) !=-np.inf, 'given beta and G gives infinity'
        else:
            assert all((-np.log(1-G/G_inf)/beta)) !=np.inf and all((-np.log(1-G/G_inf)/beta)) !=-np.inf, 'given beta and G gives infinity'
        return (-np.log(1-G/G_inf)/beta) # inverse formula

    def Preset_beta(self, G_,beta=None):
        if beta is None:
            beta=self.beta_reset
        G = np.clip(G_, 0, 1)
        G_inf = 1-1/(1-np.exp(-beta*self.N))
        if type(G) in [np.float64,float,int]:
            assert self.N -(-np.log(1-(1-G)/(1-G_inf))/beta) !=np.inf and self.N -(-np.log(1-(1-G)/(1-G_inf))/beta) !=-np.inf,'given beta and G gives infinity'
        else:
            assert all(self.N -(-np.log(1-(1-G)/(1-G_inf))/beta)) !=np.inf and all(self.N -(-np.log(1-(1-G)/(1-G_inf))/beta)) !=-np.inf,'given beta and G gives infinity'
        return self.N -(-np.log(1-(1-G)/(1-G_inf))/beta)
    #Calc Number of Pulses between two conductances
    def n_pulses_beta(self,G_i,G_f):
        """
        Get delta pulse number from current and new conductance values.
        """
        G_i = np.clip(G_i, 0, 1)
        G_f = np.clip(G_f, 0, 1)
        if G_f > G_i:
            return self.Pset_beta(G_f,self.beta_set) - self.Pset_beta(G_i,self.beta_set)
        else:
            return self.Preset_beta(G_f,self.beta_reset) - self.Preset_beta(G_i,self.beta_reset)

    #-----------------------------
    # Nonlinear function with alpha and beta

    def Pset_alpha_beta(self, G_,alpha=None,beta=None):
        if alpha is None:
            alpha=self.alpha_set
        if beta is None:
            beta=self.beta_set
        G = np.clip(G_, 0, 1)
        G_inf = 1/(1-np.exp(-beta*self.N**alpha))
        return (-np.log(1-G/G_inf)/self.beta)**(1/self.alpha)  # inverse formula
        # self.Greset = lambda p: 1 - (1 - np.exp(-(self.N - p) * self.beta_reset)) / (1 - np.exp(-self.N * self.beta_reset))
    def Preset_alpha_beta(self, G_,alpha=None,beta=None):
        if alpha is None:
            alpha=self.alpha_reset
        if beta is None:
            beta=self.beta_reset
        G = np.clip(G_, 0, 1)
        G_inf = 1-1/(1-np.exp(-beta*self.N**alpha))
        return self.N -(-np.log(1-(1-G)/(1-G_inf))/beta)**(1/alpha)
    def Gset_alpha_beta(self, p_,alpha=None,beta=None):
        if alpha is None:
            alpha=self.alpha_set
        if beta is None:
            beta=self.beta_set
        p = np.clip(p_, 0, self.N)
        return (1 - np.exp(-p**alpha * beta)) / (1 - np.exp(-self.N**alpha * beta))
    def Greset_alpha_beta(self, p_,alpha=None,beta=None):
        if alpha is None:
            alpha=self.alpha_reset
        if beta is None:
            beta=self.beta_reset
        p = np.clip(p_, 0, self.N)
        return 1 - (1 - np.exp(-(self.N - p)**alpha * beta)) / (1 - np.exp(-self.N**alpha * beta))
        
    #-----------------------------
    #Flavios original model const, linear and exponential term
    def Lin_Exp_Set2(self, p_,alpha=None,beta=None,gamma=None):
        if alpha is None:
            alpha=self.alpha_set
        if beta is None:
            beta=self.beta_set
        if gamma is None:
            gamma=self.gamma_set
        p = np.clip(p_, 0, self.N)
        return (1 - ((1-gamma) + alpha*p +gamma*np.exp(-p * beta))) / (1 - ((1-gamma) + alpha*self.N + gamma*np.exp(-self.N * beta)))
    
    def Lin_Exp_Reset2(self, p_,alpha=None,beta=None,gamma=None):
        if alpha is None:
            alpha=self.alpha_reset
        if beta is None:
            beta=self.beta_reset
        if gamma is None:
            gamma=self.gamma_reset
        p = np.clip(p_, 0, self.N)
        return 1 - (1 - ((1-gamma) + alpha*(self.N - p) +gamma*np.exp(-(self.N - p) * beta))) / (1 - ((1-gamma) + alpha*self.N + gamma*np.exp(-self.N * beta)))
    # -------------------------------------------------
        # Weight Update Functions on Software or Hardware Memristor
    # -------------------------------------------------  
    
    # Software Memristor
    def update_syn_w_emulated(self, current_w, update_val, update_type='model',noise_type=None, pulse_calc='model'):
        """
        Computes number of pulses needed for the update given by update_val, starting from the current conductance value
        current_w.
        :param current_w: current conductance or weight value
        :param update_val: desired update (delta w)
        :delta_pulse: number of pulses to be used
        """

        ideal_new_weight=np.clip(current_w + update_val,0,1)
        if ideal_new_weight>1:
            print('clipped weight is >1',ideal_new_weight)

        # positive update
        if update_val > 0.0 and current_w < 1.0:
            if pulse_calc == 'linear':
                #If pulse update curve is linear 
                current_pulse = self.N*current_w
                new_pulse_lin = self.N*(current_w + update_val)
                delta_pulse = np.round(new_pulse_lin - current_pulse) #=self.N*update_val
               
            elif pulse_calc == 'model':
                current_pulse = self.Lin_Exp_Set_INV(current_w)
                new_pulse = self.Lin_Exp_Set_INV(current_w + update_val)
                delta_pulse = np.round(new_pulse - current_pulse)

            #1.) Update Without Noise
            if update_type=='ideal': #Ideal update
                new_weight = ideal_new_weight
            elif update_type=='model': #Update on model
                current_pulse = self.Lin_Exp_Set_INV(current_w)
                new_weight = self.Lin_Exp_Set(current_pulse + delta_pulse) 
            elif update_type=='random cycle': #Update on drawn model of cycle
                cycle_max = len(self.std_noise_set_cycle)
                if (self.sign_counter == 0) and (self.cycle == None):
                    self.cycle=np.random.randint(0, cycle_max-1) # draw random cycle (only once)
                    self.sign_counter += 1
                elif (self.update_sign * delta_pulse < 0):
                    #Counter is at 2, thus new cycle needs to be drawn
                    if (self.sign_counter == 2) and (self.cycle>0) and (self.cycle<cycle_max-1):
                        self.sign_counter = 1 #Reset counter
                        if (self.count_direction == 'up'):
                            self.cycle = self.cycle + 1
                        elif (self.count_direction == 'down'):
                            self.cycle = self.cycle - 1
                    elif (self.sign_counter == 2): #If cycle is at boundary, flip direction
                        self.sign_counter = 1 #Reset counter
                        if (self.count_direction == 'up'): 
                            self.count_direction = 'down'
                            self.cycle = self.cycle - 1
                        elif (self.count_direction == 'down'): 
                            self.count_direction = 'up'
                            self.cycle = self.cycle + 1
                    else: #There is a sign change, but counter is not yet at 2
                        self.sign_counter += 1
                    
                alpha_cycle=self.alpha_noise_set[self.cycle]
                beta_cycle=self.beta_noise_set[self.cycle]
                gamma_cycle=self.gamma_noise_set[self.cycle]
                zeta_cycle=self.zeta_noise_set[self.cycle]
                std_cycle=self.std_noise_set_cycle[self.cycle]
                current_pulse_cycle_model=self.Lin_Exp_Set_4params_INV(current_w, alpha=alpha_cycle, beta=beta_cycle,gamma=gamma_cycle, zeta=zeta_cycle,clip='Glim') #Get current pulse p0 for cycle model instead
                new_weight = self.Lin_Exp_Set_4params(current_pulse_cycle_model + delta_pulse,alpha=alpha_cycle,beta=beta_cycle,gamma=gamma_cycle,zeta=zeta_cycle,clip=False) #Get new weight from update on drawn model for cycle, do not clip to be closer to memristor
            
            self.update_sign = 1

            #2.) Add Noise if desired
            if noise_type=='constant': # Add noise with constand std. dev.
                new_weight = np.clip(np.random.normal(new_weight,self.std),0,1)
            elif noise_type=='set reset separate': # Add noise for set and reset reparate
                new_weight = np.clip(np.random.normal(new_weight,self.std_noise_set),0,1)
            elif noise_type=='cycle noise': # Add noise from drawn cycle
                assert update_type=='random cycle', 'update_type must be "random cycle" to use cycle noise'
                new_weight = np.clip(np.random.normal(new_weight,std_cycle),0,1) 
            else:
                new_weight=np.clip(new_weight,0,1)
        # negative update (does not update negatively if current_w is already at zero - no negative weights)
        elif update_val < 0.0 and current_w > 0.0:
            if pulse_calc == 'linear':
                #If pulse update curve is linear                             
                current_pulse = self.N*current_w
                new_pulse_lin = self.N*(current_w + update_val)
                delta_pulse = np.round(new_pulse_lin - current_pulse)
               
            elif pulse_calc == 'model':
                current_pulse = self.Lin_Exp_Reset_INV(current_w)
                new_pulse = self.Lin_Exp_Reset_INV(current_w + update_val)
                delta_pulse = np.round(new_pulse - current_pulse)

            #1.) Update Without Noise
            if update_type=='ideal': #Ideal update
                new_weight = ideal_new_weight
            elif update_type=='model': #Update on model
                current_pulse = self.Lin_Exp_Reset_INV(current_w)
                new_weight = self.Lin_Exp_Reset(current_pulse + delta_pulse) 
            elif update_type=='random cycle': #Update on drawn model of cycle
                cycle_max = len(self.std_noise_reset_cycle)
                if (self.sign_counter == 0) and (self.cycle == None):
                    self.cycle=np.random.randint(0, cycle_max-1) # draw random cycle (only once)
                    self.sign_counter += 1
                elif (self.update_sign * delta_pulse < 0):
                    #Counter is at 2, thus new cycle needs to be drawn
                    if (self.sign_counter == 2) and (self.cycle>0) and (self.cycle<cycle_max-1):
                        self.sign_counter = 1 #Reset counter
                        if (self.count_direction == 'up'):
                            self.cycle = self.cycle + 1
                        elif (self.count_direction == 'down'):
                            self.cycle = self.cycle - 1
                    elif (self.sign_counter == 2): #If cycle is at boundary, flip direction
                        self.sign_counter = 1 #Reset counter
                        if (self.count_direction == 'up'): 
                            self.count_direction = 'down'
                            self.cycle = self.cycle - 1
                        elif (self.count_direction == 'down'): 
                            self.count_direction = 'up'
                            self.cycle = self.cycle + 1
                    else: #There is a sign change, but counter is not yet at 2
                        self.sign_counter += 1
                    
                alpha_cycle=self.alpha_noise_reset[self.cycle]
                beta_cycle=self.beta_noise_reset[self.cycle]
                gamma_cycle=self.gamma_noise_reset[self.cycle]
                zeta_cycle=self.zeta_noise_reset[self.cycle]
                std_cycle=self.std_noise_reset_cycle[self.cycle]
                current_pulse_cycle_model=self.Lin_Exp_Reset_4params_INV(current_w, alpha=alpha_cycle, beta=beta_cycle,gamma=gamma_cycle, zeta=zeta_cycle,clip='Glim') #Get current pulse p0 for cycle model instead
                new_weight = self.Lin_Exp_Reset_4params(current_pulse_cycle_model + delta_pulse,alpha=alpha_cycle,beta=beta_cycle,gamma=gamma_cycle,zeta=zeta_cycle,clip=False) #Get new weight from update on drawn model for cycle
            
            self.update_sign = -1

            #2.) Add Noise if desired
            if noise_type=='constant': # Add noise with constand std. dev.
                new_weight = np.clip(np.random.normal(new_weight,self.std),0,1)
            elif noise_type=='set reset separate': # Add noise for set and reset reparate
                new_weight = np.clip(np.random.normal(new_weight,self.std_noise_reset),0,1)
            elif noise_type=='cycle noise': # Add noise from drawn cycle
                assert update_type=='random cycle', 'update_type must be "random cycle" to use cycle noise'
                new_weight = np.clip(np.random.normal(new_weight,std_cycle),0,1)
            else:
                new_weight=np.clip(new_weight,0,1)
        elif update_val==0.0 or update_val==-0.0: #new to reduce unnessesary function calls
            new_weight=current_w
            delta_pulse=0
        elif (current_w==1 and update_val>0) or (current_w==0 and update_val<0): 
            new_weight=current_w
            delta_pulse=0
        else: #if something wrong
            print('error?:','w',current_w,'delta w',update_val)
            return 0.0,0.0,
        
        return new_weight, delta_pulse
    
    def update_emulated_no_self_correction(self, current_w, current_w_actual, update_val, update_type='model',noise_type=None, pulse_calc='model'):
        """
        Computes number of pulses needed for the update given by update_val, starting from the current conductance value
        current_w.
        :param current_w: current conductance or weight value
        :param update_val: desired update (delta w)
        :delta_pulse: number of pulses to be used
        """

        #----Reference update (assume memristor is perfect)-----
        new_weight_ref=np.clip(current_w + update_val,0,1)
        if new_weight_ref>1:
            print('clipped weight is >1',ideal_new_weight)

        #---Actual Update of Memristor Emulation----
        ideal_new_weight=np.clip(current_w_actual + update_val,0,1)
        if ideal_new_weight>1:
            print('clipped weight is >1',ideal_new_weight)
        # positive update
        if update_val > 0.0 and current_w_actual < 1.0:
            if pulse_calc == 'linear':
                #If pulse update curve is linear 
                current_pulse = self.Lin_Exp_Set_INV(current_w_actual)
                current_pulse_lin = self.N*current_w_actual
                new_pulse_lin = self.N*(current_w_actual + update_val)
                delta_pulse = np.round(new_pulse_lin - current_pulse_lin) #=self.N*update_val
                
            elif pulse_calc == 'model':
                current_pulse = self.Lin_Exp_Set_INV(current_w_actual)
                new_pulse = self.Lin_Exp_Set_INV(current_w_actual + update_val)
                delta_pulse = np.round(new_pulse - current_pulse)

            #1.) Update Without Noise
            if update_type=='ideal': #Ideal update
                new_weight = ideal_new_weight
            elif update_type=='model': #Update on model
                current_pulse = self.Lin_Exp_Set_INV(current_w_actual)
                new_weight = self.Lin_Exp_Set(current_pulse + delta_pulse) 
            elif update_type=='random cycle': #Update on drawn model of cycle
                cycle_max = len(self.std_noise_set_cycle)
                if (self.sign_counter == 0) and (self.cycle == None):
                    self.cycle=np.random.randint(0, cycle_max-1) # draw random cycle (only once)
                    self.sign_counter += 1
                elif (self.update_sign * delta_pulse < 0):
                    #Counter is at 2, thus new cycle needs to be drawn
                    if (self.sign_counter == 2) and (self.cycle>0) and (self.cycle<cycle_max-1):
                        self.sign_counter = 1 #Reset counter
                        if (self.count_direction == 'up'):
                            self.cycle = self.cycle + 1
                        elif (self.count_direction == 'down'):
                            self.cycle = self.cycle - 1
                    elif (self.sign_counter == 2): #If cycle is at boundary, flip direction
                        self.sign_counter = 1 #Reset counter
                        if (self.count_direction == 'up'): 
                            self.count_direction = 'down'
                            self.cycle = self.cycle - 1
                        elif (self.count_direction == 'down'): 
                            self.count_direction = 'up'
                            self.cycle = self.cycle + 1
                    else: #There is a sign change, but counter is not yet at 2
                        self.sign_counter += 1
                    
                alpha_cycle=self.alpha_noise_set[self.cycle]
                beta_cycle=self.beta_noise_set[self.cycle]
                gamma_cycle=self.gamma_noise_set[self.cycle]
                zeta_cycle=self.zeta_noise_set[self.cycle]
                std_cycle=self.std_noise_set_cycle[self.cycle]
                current_pulse_cycle_model=self.Lin_Exp_Set_4params_INV(current_w_actual, alpha=alpha_cycle, beta=beta_cycle,gamma=gamma_cycle, zeta=zeta_cycle,clip='Glim') #Get current pulse p0 for cycle model instead
                new_weight = self.Lin_Exp_Set_4params(current_pulse_cycle_model + delta_pulse,alpha=alpha_cycle,beta=beta_cycle,gamma=gamma_cycle,zeta=zeta_cycle,clip=False) #Get new weight from update on drawn model for cycle, do not clip to be closer to memristor
            
            self.update_sign = 1

            #2.) Add Noise if desired
            if noise_type=='constant': # Add noise with constand std. dev.
                new_weight = np.clip(np.random.normal(new_weight,self.std),0,1)
            elif noise_type=='set reset separate': # Add noise for set and reset reparate
                new_weight = np.clip(np.random.normal(new_weight,self.std_noise_set),0,1)
            elif noise_type=='cycle noise': # Add noise from drawn cycle
                assert update_type=='random cycle', 'update_type must be "random cycle" to use cycle noise'
                new_weight = np.clip(np.random.normal(new_weight,std_cycle),0,1)                
            else:
                new_weight=np.clip(new_weight,0,1)
        # negative update (does not update negatively if current_w_actual is already at zero - no negative weights)
        elif update_val < 0.0 and current_w_actual > 0.0:
            if pulse_calc == 'linear':
                #If pulse update curve is linear                             
                current_pulse = self.Lin_Exp_Reset_INV(current_w_actual)
                current_pulse_lin = self.N*current_w_actual
                new_pulse_lin = self.N*(current_w_actual + update_val)
                delta_pulse = np.round(new_pulse_lin - current_pulse_lin)
                
            elif pulse_calc == 'model':
                current_pulse = self.Lin_Exp_Reset_INV(current_w_actual)
                new_pulse = self.Lin_Exp_Reset_INV(current_w_actual + update_val)
                delta_pulse = np.round(new_pulse - current_pulse)

            #1.) Update Without Noise
            if update_type=='ideal': #Ideal update
                new_weight = ideal_new_weight
            elif update_type=='model': #Update on model
                current_pulse = self.Lin_Exp_Reset_INV(current_w_actual)
                new_weight = self.Lin_Exp_Reset(current_pulse + delta_pulse) 
            elif update_type=='random cycle': #Update on drawn model of cycle
                cycle_max = len(self.std_noise_reset_cycle)
                if (self.sign_counter == 0) and (self.cycle == None):
                    self.cycle=np.random.randint(0, cycle_max-1) # draw random cycle (only once)
                    self.sign_counter += 1
                elif (self.update_sign * delta_pulse < 0):
                    #Counter is at 2, thus new cycle needs to be drawn
                    if (self.sign_counter == 2) and (self.cycle>0) and (self.cycle<cycle_max-1):
                        self.sign_counter = 1 #Reset counter
                        if (self.count_direction == 'up'):
                            self.cycle = self.cycle + 1
                        elif (self.count_direction == 'down'):
                            self.cycle = self.cycle - 1
                    elif (self.sign_counter == 2): #If cycle is at boundary, flip direction
                        self.sign_counter = 1 #Reset counter
                        if (self.count_direction == 'up'): 
                            self.count_direction = 'down'
                            self.cycle = self.cycle - 1
                        elif (self.count_direction == 'down'): 
                            self.count_direction = 'up'
                            self.cycle = self.cycle + 1
                    else: #There is a sign change, but counter is not yet at 2
                        self.sign_counter += 1
                    
                alpha_cycle=self.alpha_noise_reset[self.cycle]
                beta_cycle=self.beta_noise_reset[self.cycle]
                gamma_cycle=self.gamma_noise_reset[self.cycle]
                zeta_cycle=self.zeta_noise_reset[self.cycle]
                std_cycle=self.std_noise_reset_cycle[self.cycle]
                current_pulse_cycle_model=self.Lin_Exp_Reset_4params_INV(current_w_actual, alpha=alpha_cycle, beta=beta_cycle,gamma=gamma_cycle, zeta=zeta_cycle,clip='Glim') #Get current pulse p0 for cycle model instead
                new_weight = self.Lin_Exp_Reset_4params(current_pulse_cycle_model + delta_pulse,alpha=alpha_cycle,beta=beta_cycle,gamma=gamma_cycle,zeta=zeta_cycle,clip=False) #Get new weight from update on drawn model for cycle
            
            self.update_sign = -1

            #2.) Add Noise if desired
            if noise_type=='constant': # Add noise with constand std. dev.
                new_weight = np.clip(np.random.normal(new_weight,self.std),0,1)
            elif noise_type=='set reset separate': # Add noise for set and reset reparate
                new_weight = np.clip(np.random.normal(new_weight,self.std_noise_reset),0,1)
            elif noise_type=='cycle noise': # Add noise from drawn cycle
                assert update_type=='random cycle', 'update_type must be "random cycle" to use cycle noise'
                new_weight = np.clip(np.random.normal(new_weight,std_cycle),0,1) 
            else:
                new_weight=np.clip(new_weight,0,1)
        elif update_val==0.0 or update_val==-0.0: #new to reduce unnessesary function calls
            new_weight=current_w_actual
            delta_pulse=0
        elif (current_w_actual==1 and update_val>0) or (current_w_actual==0 and update_val<0): #new
            new_weight=current_w_actual
            delta_pulse=0
        else: #if something wrong
            print('error?:','w',current_w_actual,'delta w',update_val)
            return 0.0,0.0,0
        
        return new_weight_ref, new_weight, delta_pulse
    
    # Hardware Memristor
    def update_syn_w_memristor(self, current_w, update_val, pulse_calc='model'):
        if current_w+update_val>1 or current_w+update_val<0:
            print('update would go beyond 0,1 range','current w',current_w,'delta w',update_val)
        if (current_w==1 and update_val>0) or (current_w==0 and update_val<0):
            delta_pulse=0
        elif pulse_calc == 'linear':
            delta_pulse=self.write_linear(G_f=current_w+update_val,G_i=current_w,print_p=False)
        elif pulse_calc == 'model':
            delta_pulse=self.write(G_f=current_w+update_val,G_i=current_w,print_p=False)
        
        if abs(delta_pulse)>0: #if there was an update
            new_weight = np.clip(self.read(print_G=False,plot=False), 0, 1) #ploooot
            print(f'w(t):{current_w:17.3f} w(t+1)_ideal:{current_w+update_val:11.3f} w(t+1):{new_weight:11.3f} delta_w:{update_val:11.3f} delta_p:{int(delta_pulse)} w_error:{new_weight-(current_w+update_val):11.3f}')
        else:
            new_weight = current_w
        

        return new_weight, delta_pulse

    # -------------------------------------------------
        # Measurement and Helper Functions
    # ------------------------------------------------- 
    def characterize(self,repeats,plot=False,fit=True,save=True,chipname='chip',ID='ID',savepath='Z:/Experiments/DefaultSavePath'):
        """
        Retrieves pulses / resistance characteristic with AWG - OSC.
        """
        m=Pulse_Measurement()
        N=self.wf_write['n'][0] #assume const.

        m.set_wf_read_prepend(self.meas_dict_read['device_src'], self.wf_write, self.meas_dict_read['device_ch']) #device
        m.set_wf_read_prepend(self.meas_dict_read['ref_src'], self.wf_write, self.meas_dict_read['ref_ch']) #reference signal

        #Init arrays
        pset = np.array([]) 
        preset = np.array([]) 
        Gset=np.array([]) 
        Greset=np.array([]) 
        for i in range(repeats):
            results_AWG = m.meas_AWG(self.meas_dict_write,show_plot = plot, save=False, post_processing=True,time_measurement=False,HW_messages=True) #print for debugging
            G=results_AWG['G_mean'][np.isnan(results_AWG['G_mean'])==False] #it is nan every sample/timestep where no mean available
            pset=np.append(pset,np.arange(N+1))
            Gset=np.append(Gset,np.array(G[:N+1]))
            preset=np.append(preset,np.arange(N+1))
            Greset=np.append(Greset,np.flip(np.array(G[N:])))
    
        pset=pd.Series(pset)
        preset=pd.Series(preset)
        Gset=pd.Series(Gset)
        Greset=pd.Series(Greset)
        
        # Fit Parameters
        failed_fit=False
        if fit:
            # try:
            params = self.get_params(N,pset,Gset,preset,Greset,plot=False)
            # except:
            #     failed_fit=True
            #     print('fit failed')

        #Plot Raw Conductance Values and Normalized one with Fit
        if fit==False or failed_fit:
            fig,ax=plt.subplots()
            ax.plot(pset,1e6*Gset,'.',label='Set')
            ax.plot(preset,1e6*Greset,'.',label='Reset')
            ax.legend()
            ax.set_xlabel('#Pulses')
            ax.set_ylabel('Conductance ($\mu S$)')
            plt.show()
        else:
            fig,axs=plt.subplots(2,1,sharex=True)
            ax=axs[0]
            ax.plot(pset,1e6*Gset,'.',label='Set')
            ax.plot(preset,1e6*Greset,'.',label='Reset')
            ax.legend()
            ax.set_ylabel('Conductance ($\mu S$)')

            ax=axs[1]
            x=np.arange(N)
            ax.plot(pset,params['Gset_norm'],'.')
            ax.plot(preset,params['Greset_norm'],'.')
            ax.plot(x,self.Lin_Exp_Set(x, self.alpha_set, self.beta_set), '-b',label='Set')
            ax.plot(x,self.Lin_Exp_Reset(x, self.alpha_reset, self.beta_reset), '-r',label='Reset')
            ax.legend()
            ax.set_xlabel('#Pulses')
            ax.set_ylabel('Normalized Conductance')
            plt.show()

        results={
            'pset': pset.tolist(),
            'preset': preset.tolist(),
            'Gset': Gset.tolist(),
            'Greset': Greset.tolist(),
            'repeats':repeats
        }
        if failed_fit==False and fit==True:
            save_dict = {
                    **self.wf_write,
                    **self.meas_dict_write,
                    **results,
                    **params
                }
        else:
            save_dict = {
                **self.wf_write,
                **self.meas_dict_write,
                **results,
            }
        if save:
            m.save_meas_json(meas_dict=save_dict,fig=fig,key='AWG',chip=chipname,ID=ID,path=savepath)

        return results
        
    
    def get_params(self,N,pset,Gset,preset,Greset,plot=True):
        """
        Retrieves set and reset functions' parameters from pulses / resistance characteristic returned by characterize.
        """
        pset=pd.Series(pset)
        preset=pd.Series(preset)
        Gset=pd.Series(Gset)
        Greset=pd.Series(Greset)

        self.N=N

        # Get Limits of Conductance
        self.G0=Gset[pset==0].mean()
        self.G1=Gset[pset==N].mean()

        #Subtract G0
        Gset=(Gset-self.G0)
        Greset=(Greset-self.G0)

        #Normalize
        Gset_norm=Gset/(self.G1-self.G0)
        Greset_norm=Greset/(self.G1-self.G0)

        #Fit 
        popt_s, pcov_s = curve_fit(self.Lin_Exp_Set, pset, Gset_norm,p0=[-0.005,0.05], maxfev=100000) 
        popt_r, pcov_r = curve_fit(self.Lin_Exp_Reset, preset, Greset_norm,p0=[-0.005,0.05], maxfev=100000)

        self.alpha_set=popt_s[0]
        self.beta_set=popt_s[1]
        self.alpha_reset=popt_r[0]
        self.beta_reset=popt_r[1]
        
        if plot:
            x=np.arange(N)
            fig,ax=plt.subplots()
            ax.plot(pset,Gset_norm,'.')
            ax.plot(preset,Greset_norm,'.')
            ax.plot(x,self.Lin_Exp_Set(x,self.alpha_set, self.beta_set), '-b',label='Set')
            ax.plot(x,self.Lin_Exp_Reset(x,self.alpha_reset, self.beta_reset), '-r',label='Reset')
            ax.legend()
            ax.set_xlabel('#Pulses')
            ax.set_ylabel('Normalized Conductance')
            plt.show()
        
        params = {
            'N' : N,
            'G0' : self.G0,
            'G1' : self.G1,
            'Gset_norm':Gset_norm.tolist(),
            'Greset_norm':Greset_norm.tolist(),
            'alpha_set':self.alpha_set,
            'beta_set' :self.beta_set,
            'alpha_reset':self.alpha_reset,
            'beta_reset' :self.beta_reset,        }
        return params 
    
    def get_params_2(self,N,pset,Gset,preset,Greset,plot=True):
        """
        Retrieves set and reset functions' parameters from pulses / resistance characteristic returned by characterize.
        """
        pset=pd.Series(pset)
        preset=pd.Series(preset)
        Gset=pd.Series(Gset)
        Greset=pd.Series(Greset)

        self.N=N

        # Get Limits of Conductance
        self.G0=Gset[pset==0].mean()
        self.G1=Gset[pset==N].mean()

        #Subtract G0
        Gset=(Gset-self.G0)
        Greset=(Greset-self.G0)

        #Normalize
        Gset_norm=Gset/(self.G1-self.G0)
        Greset_norm=Greset/(self.G1-self.G0)

        #Fit
        popt_s, pcov_s = curve_fit(self.Gset_beta, pset, Gset_norm,p0=0.03,bounds=(0, 1)) #fit beta in range [0,1] (enough) start guess 0
        popt_r, pcov_r = curve_fit(self.Greset_beta, preset, Greset_norm,p0=0.03,bounds=(0, 1)) #fit beta in range [0,1] (enough) start guess 0

        self.beta_set=popt_s[0]
        self.beta_reset=popt_r[0]

        #set not used params to 0
        self.alpha_set=0
        self.alpha_reset=0

        if plot:
            x=np.arange(N)
            fig,ax=plt.subplots()
            ax.plot(pset,Gset_norm,'.')
            ax.plot(preset,Greset_norm,'.')
            ax.plot(x,self.Gset_beta(x, self.beta_set), '-b',label='Set')
            ax.plot(x,self.Greset_beta(x, self.beta_reset), '-r',label='Reset')
            ax.legend()
            ax.set_xlabel('#Pulses')
            ax.set_ylabel('Normalized Conductance')
            plt.show()
        
        params = {
            'N' : N,
            'G0' : self.G0,
            'G1' : self.G1,
            'Gset_norm':Gset_norm.tolist(),
            'Greset_norm':Greset_norm.tolist(),
            'beta_set' :self.beta_set,
            'beta_reset':self.beta_reset
        }
        return params
    
    def get_params_3(self,N,pset,Gset,preset,Greset,plot=True):
            """
            Retrieves set and reset functions' parameters from pulses / resistance characteristic returned by characterize.
            """
            pset=pd.Series(pset)
            preset=pd.Series(preset)
            Gset=pd.Series(Gset)
            Greset=pd.Series(Greset)

            self.N=N

            # Get Limits of Conductance
            self.G0=Gset[pset==0].mean()
            self.G1=Gset[pset==N].mean()

            #Subtract G0
            Gset=(Gset-self.G0)
            Greset=(Greset-self.G0)

            #Normalize
            Gset_norm=Gset/(self.G1-self.G0)
            Greset_norm=Greset/(self.G1-self.G0)

            #Fit
            popt_s, pcov_s = curve_fit(self.Gset_alpha_beta, pset, Gset_norm,p0=[0.4,0.002]) 
            popt_r, pcov_r = curve_fit(self.Greset_alpha_beta, preset, Greset_norm,p0=[0.4,0.002])

            self.alpha_set=popt_s[0]
            self.beta_set=popt_s[1]
            self.alpha_reset=popt_r[0]
            self.beta_reset=popt_r[1]
            
            if plot:
                x=np.arange(N)
                fig,ax=plt.subplots()
                ax.plot(pset,Gset_norm,'.')
                ax.plot(preset,Greset_norm,'.')
                ax.plot(x,self.Gset_alpha_beta(x,self.alpha_set, self.beta_set), '-b',label='Set')
                ax.plot(x,self.Greset_alpha_beta(x,self.alpha_reset, self.beta_reset), '-r',label='Reset')
                ax.legend()
                ax.set_xlabel('#Pulses')
                ax.set_ylabel('Normalized Conductance')
                plt.show()
            
            params = {
                'N' : N,
                'G0' : self.G0,
                'G1' : self.G1,
                'Gset_norm':Gset_norm.tolist(),
                'Greset_norm':Greset_norm.tolist(),
                'alpha_set':self.alpha_set,
                'beta_set' :self.beta_set,
                'alpha_reset':self.alpha_reset,
                'beta_reset' :self.beta_reset,        }
            return params 
    def get_params_4(self,N,pset,Gset,preset,Greset,plot=True):
            """
            Retrieves set and reset functions' parameters from pulses / resistance characteristic returned by characterize.
            """
            pset=pd.Series(pset)
            preset=pd.Series(preset)
            Gset=pd.Series(Gset)
            Greset=pd.Series(Greset)

            self.N=N

            # Get Limits of Conductance
            self.G0=Gset[pset==0].mean()
            self.G1=Gset[pset==N].mean()

            #Subtract G0
            Gset=(Gset-self.G0)
            Greset=(Greset-self.G0)

            #Normalize
            Gset_norm=Gset/(self.G1-self.G0)
            Greset_norm=Greset/(self.G1-self.G0)

            #Fit
            popt_s, pcov_s = curve_fit(self.Lin_Exp_Set2, pset, Gset_norm,p0=[-0.005,0.05,1]) 
            popt_r, pcov_r = curve_fit(self.Lin_Exp_Reset2, preset, Greset_norm,p0=[-0.005,0.05,1])

            self.alpha_set=popt_s[0]
            self.beta_set=popt_s[1]
            self.gamma_set=popt_s[2]
            print("alpha s: ", self.alpha_set, "beta s: ", self.beta_set, "gamma s: ", self.gamma_set)

            self.alpha_reset=popt_r[0]
            self.beta_reset=popt_r[1]
            self.gamma_reset=popt_r[2]
            print("alpha r: ", self.alpha_reset, "beta r: ", self.beta_reset, "gamma r: ", self.gamma_reset)

            if plot:
                x=np.arange(N)
                fig,ax=plt.subplots()
                ax.plot(pset,Gset_norm,'.')
                ax.plot(preset,Greset_norm,'.')
                ax.plot(x,self.Lin_Exp_Set2(x,self.alpha_set, self.beta_set,self.gamma_set), '-b',label='Set')
                ax.plot(x,self.Lin_Exp_Reset2(x,self.alpha_reset, self.beta_reset,self.gamma_reset), '-r',label='Reset')
                ax.legend()
                ax.set_xlabel('#Pulses')
                ax.set_ylabel('Normalized Conductance')
                plt.show()
            
            params = {
                'N' : N,
                'G0' : self.G0,
                'G1' : self.G1,
                'Gset_norm':Gset_norm.tolist(),
                'Greset_norm':Greset_norm.tolist(),
                'alpha_set':self.alpha_set,
                'beta_set' :self.beta_set,
                'alpha_reset':self.alpha_reset,
                'beta_reset' :self.beta_reset,        }
            return params 
    
    def get_params_3param(self,N,pset,Gset,preset,Greset,plot=True):
        """
        Retrieves set and reset functions' parameters from pulses / resistance characteristic returned by characterize.
        """
        pset=pd.Series(pset)
        preset=pd.Series(preset)
        Gset=pd.Series(Gset)
        Greset=pd.Series(Greset)

        self.N=N

        # Get Limits of Conductance
        self.G0=Gset[pset==0].mean()
        self.G1=Gset[pset==N].mean()

        #Subtract G0
        Gset=(Gset-self.G0)
        Greset=(Greset-self.G0)

        #Normalize
        Gset_norm=Gset/(self.G1-self.G0)
        Greset_norm=Greset/(self.G1-self.G0)

        #Fit
        popt_s, pcov_s = curve_fit(self.Lin_Exp_Set_3params, pset, Gset_norm,p0=[-0.005,0.05, 0.1]) 
        popt_r, pcov_r = curve_fit(self.Lin_Exp_Reset_3params, preset, Greset_norm,p0=[-0.005,0.05, 0.1])

        self.alpha_set=popt_s[0]
        self.beta_set=popt_s[1]
        self.gamma_set=popt_s[2]
        print("alpha s: ", self.alpha_set, "beta s: ", self.beta_set, "gamma s: ", self.gamma_set)

        self.alpha_reset=popt_r[0]
        self.beta_reset=popt_r[1]
        self.gamma_reset=popt_r[2]
        print("alpha r: ", self.alpha_reset, "beta r: ", self.beta_reset, "gamma r: ", self.gamma_reset)

        if plot:
            x=np.arange(N)
            fig,ax=plt.subplots()
            ax.plot(pset,Gset_norm,'.')
            ax.plot(preset,Greset_norm,'.')
            ax.plot(x,self.Lin_Exp_Set_3params(x,self.alpha_set, self.beta_set, self.gamma_set), '-b',label='Set')
            ax.plot(x,self.Lin_Exp_Reset_3params(x,self.alpha_reset, self.beta_reset, self.gamma_reset), '-r',label='Reset')
            ax.legend()
            ax.set_xlabel('#Pulses')
            ax.set_ylabel('Normalized Conductance')
            plt.show()
        
        params = {
            'N' : N,
            'G0' : self.G0,
            'G1' : self.G1,
            'Gset_norm':Gset_norm.tolist(),
            'Greset_norm':Greset_norm.tolist(),
            'alpha_set':self.alpha_set,
            'beta_set' :self.beta_set,
            'gamma_set': self.gamma_set,
            'alpha_reset':self.alpha_reset,
            'beta_reset' :self.beta_reset,
            'gamma_reset': self.gamma_reset,        }
        return params 

    
    def read(self,plot=False,print_G=True):
        """
        Reads conductance value with AWG - OSC.
        """
        m=Pulse_Measurement()

        m.get_and_set_wf(self.meas_dict_read['device_src'], self.wf_read, self.meas_dict_read['device_ch']) #device
        m.get_and_set_wf(self.meas_dict_read['ref_src'], self.wf_read, self.meas_dict_read['ref_ch']) #reference signal

        results = m.meas_AWG(self.meas_dict_read, show_plot = plot, save=False, post_processing=True,time_measurement=False,cutoff_read=0.01,HW_messages=False)  #print for debugging
        Gread=results['G_mean'][np.isnan(results['G_mean'])==False].iloc[0] #it is nan every sample/timestep where no mean available
        if print_G:
            print('G:',Gread)
            print('Gnorm:',(Gread-self.G0)/(self.G1-self.G0))
        return (Gread-self.G0)/(self.G1-self.G0)
        # return (results['R_mean'][0]**-1/params['A'] - 1)/params['B'] * self.ss - (self.ss-1)/2 #Oscar Version with different params

    def write(self, G_f,G_i=None,print_p=True,clipped=True):
        """
        Applies pulses with AWG to go from Gi to Gf
        Clipps Gi and Gf to [0,1]
        """
        m=Pulse_Measurement()
        if clipped:
            if G_i==None:
                G_i = np.clip(self.read(print_G=False), 0, 1)
            else:
                G_i= np.clip(G_i, 0, 1)
            G_f = np.clip(G_f, 0, 1)
        else:
            if G_i==None:
                G_i = self.read(print_G=False)
        n = self.n_pulses(G_i, G_f,clipped=clipped)
        
        n=round(n)
        if abs(n)>0:
            if print_p:
                print(n,'Pulses')
            #Get pulse parameters
            if n > 0: #set
                V=self.wf_write['V'][0]
                W=self.wf_write['W'][0]
            else: #reset
                V=self.wf_write['V'][1]
                W=self.wf_write['W'][1]
            n=abs(n)
            T= self.wf_write['T'] #the same for both for now
            V_read= self.wf_write['read_V']

            wf = {
                'V' : [V_read]+[V for i in range(n)], # pulse voltages
                'n' : [1] + [1 for i in range(n)],   # pulse repetitions
                'W' : [T-W]+[W for i in range(n)],     # pulse widths
                'T' : [T-W]+[T for i in range(n)],     # pulse period
                'read_offset' : True,
                'read_V' : V_read,
                'spp':self.wf_write['spp'],
                'output_impedance': self.wf_write['output_impedance']
            }

            #set wf to correct AWG and channel
            device_src= self.meas_dict_read['device_src']
            device_ch= self.meas_dict_read['device_ch']

            m.get_and_set_wf(device_src, wf, device_ch)

            DAQ = instruments['DAQ']

            if DAQ.connected:
                DAQ.set_conn(ch_DAQ_AWG)

            AWG = instruments[device_src]
            
            AWG.set_outp(device_ch, 1)
            AWG.trigger()
            AWG.set_outp(device_ch, 0)
        return n
        
    def write_linear(self, G_f,G_i=None,print_p=True):
        """
        Writes conductance value G with AWG.
        Clipps Gi and Gf to [0,1]
        """
        m=Pulse_Measurement()
        if G_i==None:
            G_i = np.clip(self.read(print_G=False), 0, 1)
        else:
            G_i= np.clip(G_i, 0, 1)
        G_f = np.clip(G_f, 0, 1)
        n = self.n_pulses_lin(G_i, G_f)
        
        #Constant Voltage Pulses
        n=round(n)
        if abs(n)>0:
            if print_p:
                print(n,'Pulses')
            #Get pulse parameters
            if n > 0: #set
                V=self.wf_write['V'][0]
                W=self.wf_write['W'][0]
            else: #reset
                V=self.wf_write['V'][1]
                W=self.wf_write['W'][1]
            n=abs(n)
            T= self.wf_write['T'] #the same for both for now
            V_read= self.wf_write['read_V']

            #Create measurement dict in notation of new Code
            wf = {
                'V' : [V_read]+[V for i in range(n)], # pulse voltages
                'n' : [1] + [1 for i in range(n)],   # pulse repetitions
                'W' : [T-W]+[W for i in range(n)],     # pulse widths
                'T' : [T-W]+[T for i in range(n)],     # pulse period
                'read_offset' : True,
                'read_V' : V_read,
                'spp':self.wf_write['spp'],
                'output_impedance': self.wf_write['output_impedance']
            }

            #set wf to correct AWG and channel
            device_src= self.meas_dict_read['device_src']
            device_ch= self.meas_dict_read['device_ch']

            m.get_and_set_wf(device_src, wf, device_ch)

            DAQ = instruments['DAQ']

            if DAQ.connected:
                DAQ.set_conn(ch_DAQ_AWG)

            AWG = instruments[device_src]
            
            AWG.set_outp(device_ch, 1)
            AWG.trigger()
            AWG.set_outp(device_ch, 0)
        return n
