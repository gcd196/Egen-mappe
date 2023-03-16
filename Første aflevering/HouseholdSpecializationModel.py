
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5

        # c. household production
        par.alpha = 0.5
        par.alpha_vec=(0.25,0.5,0.75)
        par.sigma = 1.0
        par.sigma_vec=(0.5,1.0,1.5)

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5) #Vector for the female wage. Can be useful. 

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size) #Hours worked by men
        sol.HM_vec = np.zeros(par.wF_vec.size) #Working at home by men
        sol.LF_vec = np.zeros(par.wF_vec.size) #Hours worked by women
        sol.HF_vec = np.zeros(par.wF_vec.size) #Working at home for women

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF,sigma_2,alpha_2):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production #Changed 16. march. based on Jonas' formula
        if sigma_2 == 0:
            H=np.fmin(HM,HF)
        elif sigma_2 == 1:
            H = HM**(1-alpha_2)*HF**alpha_2
        else:
            H= ((1-alpha_2)*HM**((sigma_2-1)/sigma_2)+
                alpha_2*HF**((sigma_2-1)/sigma_2))**(sigma_2/(sigma_2-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_) #Notice definition of epsilon_
        
        return utility - disutility

    def solve_discrete(self,Sigma_d, Alpha_d, do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49) #It creases 49 number evenly distributed between 0 and 24. I.e. restrictions
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations based on restrictions. It makes four vectors
    
        LM = LM.ravel() # make it a vector 1. dimensional. All combination. Each element one combination. 
        HM = HM.ravel() 
        LF = LF.ravel()
        HF = HF.ravel()


        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF,sigma_2=Sigma_d,alpha_2=Alpha_d) #Saves all values
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u) #Returns the index for the maximum of all the values. NOT VALUE. 
        
        opt.LM = LM[j] #Finds the maximum value
        opt.HM = HM[j] 
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}') #Prints value. 
        print(Sigma_d)
        print(Alpha_d)

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        pass    

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass