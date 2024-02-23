from _context import *
# sys.path.append("../../optolab_control_software")
# from meas import *
import numpy as np
from scipy.special import softmax

class ActorCritic:
    def __init__(self, n_states, n_actions, T):
        self.n_states = n_states
        
        # define network weights
        self.theta = np.zeros([n_states, n_actions])  # Actor network
        self.w = np.zeros(n_states)  # Critic network

        # policy as sampling from the softmax of an array of action values
        self.pi = lambda x: np.random.choice([0,1], p=softmax(x/T))
        
    def value(self, x):
        """
        Computes the value of the input vector x based on the critic weights w
        """
        return x @ self.w
        
    def action(self, x):
        """
        Computes the action values of the input vector x based on the actor weights \theta
        Returns both the sampled action and the logits h
        """
        h = np.dot(self.theta.T, x)
        return self.pi(h), h
        
    def one_hot(self, s):
        """
        Returns one hot encoded vector of states given the state index x
        """
        x = np.zeros(self.n_states)
        x[s] = 1
        return x
    def HW_delta_w(self,m1,m2,reward,alpha,gamma,G1_norm,G2_norm,plot=False,print_info=True): #to be implemented
        
        #Memristor Current State -> G1,G2

        #Amplifier Settings:
        gain=1e5#m1.meas_dict_read['gain_delta_w']
        AMP=instruments['DHPCA_100']

        # Used Resitor value to supply reward current and constant term for normalization
        R3=9.99e3
        GR=1/R3

        Gmax1=m1.G1
        Gmin1=m1.G0

        if m2!=None: #check if next state available
            Gmax2=m2.G1
            Gmin2=m2.G0
        else: #arbitrary values such that difference not 0
            Gmax2=2 
            Gmin2=1
        # G1_norm=(G1-Gmin1)/(Gmax1-Gmin1)
        # if m2!=None: #if no next state
        #     G2_norm=(G2-Gmin2)/(Gmax2-Gmin2)
        # else:
        #     G2_norm=0

        GR_norm=1
        if print_info:
            if m2!=None: #check if next state available
                print('Vt',G1_norm,'Vt+1',G2_norm)
            else:
                print('Vt',G1_norm)

        V1=-alpha
        if m2!=None: #check if next state available
            V2=alpha*gamma
        else:
            V2=0
        V3=alpha*reward

        Inorm=V1*G1_norm+V2*G2_norm+V3*GR_norm #expectation of result alpha*(r+gamma*v_st+1-V_st)
        if print_info:
            print('Calculated Delta w:',Inorm)

        V1_comp=V1/(Gmax1-Gmin1)/gain
        V2_comp=V2/(Gmax2-Gmin2)/gain
        constant=-V1_comp*Gmin1 -V2_comp*Gmin2
        V3_comp=V3/GR/gain +constant/GR #third term has to be =V3 resp. =alpha


        #Check if Voltages too large
        if m2!=None: #check if next state available
            Vs=[V1_comp,V2_comp,V3_comp]
        else:
            Vs=[V1_comp,V1_comp,V3_comp] #same voltage for 1 and 2 as same instr. and channel, thus 1st will be overwritten by 2nd
        if print_info:
            print('Applied Voltages',V1_comp*1e3,'mV',V2_comp*1e3,'mV',V3_comp*1e3,'mV')
        assert abs(max(Vs[:2]))<0.5, 'read Voltages are too high and could alter state, increase gain'
        assert abs(Vs[2])<=5, 'voltage to resistor >max AWG'

        m=Pulse_Measurement() #Create instance of Class (also resets all waveform params - important to excecute each time

        R_min=1e3
        R_s=0
        R_input=AMP.input_impedance[gain][0]
        fraction=gain/(R_min+R_input+R_s)
        V_expected=Inorm
        V_scale=2*V_expected/fraction #as this will be multiplied by 'fraction' in the function, factor 2 to be safe
        trigger_level=0.4*V_expected #lowered that error ok

        # Get correct instruments and channels
        # print('two memristors available',m2!=None)
        if m2!=None: #if no next state
            instruments_list= [m1.meas_dict_read['device_src'],m2.meas_dict_read['device_src'],'AWG_IIS'] #all sources
            channels=[m1.meas_dict_read['device_ch'],m2.meas_dict_read['device_ch'],2] #all channels
        else: #use same instr and ch for first two - just let it overwrite itself with same thing
            instruments_list= [m1.meas_dict_read['device_src'],m1.meas_dict_read['device_src'],'AWG_IIS'] #all sources
            channels=[m1.meas_dict_read['device_ch'],m1.meas_dict_read['device_ch'],2] #all channels

        meas_dict={ #before 1e5 OSC bandwidth
            'R_s':R_s, #series resistor in measurement
            'R_min': R_min, #min. resitance that all devices reaches during meaurement
            'V_scale': V_scale,
            'trigger_level': trigger_level,
            'OSC_trigger_ch': 2,

            'gain': gain, 
            'amp_type': 'DHPCA_100',
            'low_high_gain': 'low',

            'device_src': m1.meas_dict_read['device_src'], #main src used for data processing etc
            'device_ch': m1.meas_dict_read['device_ch'], #main channel
            'instruments': instruments_list,
            'channels': channels,
        }
        # print(meas_dict)
        # Measurement
        noise_limit=0.035
        try:
            if abs(V_expected)>noise_limit:
                I,delta_w=m.meas_crossbar_current(Vs=Vs,meas_dict=meas_dict,HIZ=True,save_plot=False,show_plot=plot, save=False)
            else:
                print('Too small current to read delta w reliably')
                delta_w=noise_limit*np.sign(V_expected) #neeew
                I=delta_w/gain #neew
        except:
            print('!!!Delta W failed!!!')
            delta_w=0
            I=0
        error=delta_w-Inorm
        #Print Results
        # print('Applied Voltages',V1_comp*1e3,'mV',V2_comp*1e3,'mV',V3_comp*1e3,'mV')
        # print('Measured Input Current Amplifier:',I)
        if print_info:
            print('Measured Delta w:',delta_w)
        if Inorm!=0:
            print('Absolute Error:',error,'Relative:',100*error/Inorm,'%') #positive error means delta w too big
        else:
            print('Absolute Error:',error) #positive error means delta w too big
        return delta_w,error
    
class ActorCriticWaterMaze:
    def __init__(self, n_states, T):
        self.n_states = n_states
        self.T = T

        s = np.sqrt(1/2)
        self.actions = np.array([[+1,  0],  # E
                                    [+s, +s],  # NE
                                    [ 0, +1],  # N
                                    [-s, +s],  # NW
                                    [-1,  0],  # W
                                    [-s, -s],  # SW
                                    [ 0, -1],  # S
                                    [+s, -s]]  # SE
                                )

        # define network weights
        self.w_v = np.zeros(n_states)  # Critic network
        self.w_a = np.zeros((n_states, len(self.actions)))  # Action up-down-left-right

        # policy as sampling from the softmax of an array of action values
        self.pi = lambda x: np.random.choice(range(len(self.actions)), p=softmax(x/self.T))

    def value(self, x):
        """
        Computes the value of the input vector x based on the critic weights w
        """
        return x @ self.w_v

    def action(self, x):
        """
        Computes the action values of the input vector x based on the actor weights
        Returns both the sampled action (NS, EW) and the logits h
        """
        h_a = np.matmul(x, self.w_a)
        sampled_a = self.pi(h_a)
        return sampled_a, h_a


