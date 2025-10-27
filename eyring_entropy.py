import numpy as np
np.testing.Tester = np.testing.TestCase
import pandas as pd
import json
import scipy
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from feos.si import * # SI numbers and constants
from feos.pcsaft import *
from feos.eos import *


class EyringEntropyModel():
    """
    ...
    """
    def __init__(self, parameters, data=[], data_diffusion=[], 
                 p=[ 2,2, 1.4, .1, .1]):
        self.p = p
        self.data = data
        self.parameters = parameters
        self.data_diffusion = data_diffusion
        self.dd = [1/6]
        if len(data_diffusion) > 0:
            self.diffusion_flag = True
        else:
            self.diffusion_flag = False
        return

    def predict_log_vf13(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        _,nf,_,_,_ = p
        n13 = 84446884.95844965
        vf13_0 =  data["V"]**(1/3) / n13
        return np.log(vf13_0) +  data["H_res"]/8.314/data["temperature"]*nf/3
        #return np.log(vf13_0) +  data["E_res"]/8.314/data["temperature"]/nf/3
        #return np.log(vf13_0) +  data["s_res"]/8.314/nf/3

    def predict_log_lambdas(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_,_ = p
        lambdas = 1/ (data["V"])
        return np.log(lambdas) - np.log(a**2)

    def predict_eyring_ref(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_,_ = p

        const = np.log(7.227814978226432) # np.sqrt(2*np.pi*8.31446261815324)
        c0 = np.log( np.sqrt(data["M"]*data["temperature"]) )

        log_vf13 = self.predict_log_vf13(p=p,data=data)
        log_lambdas = self.predict_log_lambdas(p=p,data=data)

        return const+c0+log_vf13+log_lambdas

    def predict_energy_barrier(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        _,_,n0,n1,n2 = p
        x = -data["s_res"]/8.314
        #x = -data["E_res"]/8.314/data["temperature"]
        #x = -data["H_res"]/8.314/data["temperature"]
        return n0*x + n1*x**2 + n2*x**3
    
    def predict(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        ey_ref = self.predict_eyring_ref(p=p,data=data)
        barrier = self.predict_energy_barrier(p=p,data=data)
        return ey_ref + barrier

    def predict_diffusion(self,p=[],data=[],dd=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        if len(dd)==0:
            dd=self.dd            
        log_dd = np.log(dd[0])
        log_vis = self.predict(p=p,data=data)
        n13 = 84446884.95844965
        log_vf13 =  np.log(data["V"]**(1/3) / n13)
        log_kT = np.log( 1.380649e-23*data["temperature"] )
        return log_kT -log_vis -log_vf13 + log_dd
    
    def error(self,p=[],data=[],y_key="viscosity"):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data     
        pred = self.predict(p=p,data=data)
        error = np.mean( ( pred - np.log(data[y_key]) )**2 )
        return error*1e10

    def error_diffusion(self,dd=[],data_diffusion=[],y_key_diffsion="D",p=[]):
        if len(p)==0:
            p=self.p              
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion                 
        if len(dd)==0:
            dd=self.dd        
        pred = self.predict_diffusion(p=p,data=data_diffusion,dd=dd)
        error = np.sum( ( pred - np.log(data_diffusion[y_key_diffsion]) )**2 )
        return error*1e10
    
    def error_both(self,pdd=[], data=[],y_key="viscosity",
                        data_diffusion=[],y_key_diffsion="D"):
        if len(pdd)==0:
            p=self.p 
            dd=self.dd
        else:
            p=pdd[:-1]
            dd=[pdd[-1]]
        if len(data)==0:
            data=self.data                 
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion                 
    
        pred = self.predict(p=p,data=data)
        error = np.mean( ( pred - np.log(data[y_key]) )**2 )

        pred = self.predict_diffusion(p=p,data=data_diffusion,dd=dd)
        d_error = np.mean( ( pred - np.log(data_diffusion[y_key_diffsion]) )**2 )
        error += 1.5*d_error
        return error*1e10

    def train(self,p=[],data=[],y_key="viscosity"):
        if len(p)==0:
            p=self.p       
        if len(data)==0:
            data=self.data            
        bounds = [[0.1,40],[-1.5,1.5],[0.1,8],[-1000,800],[-1000,800]]
        ferr = lambda x: self.error(x,data,y_key)
        res = minimize(ferr, p, bounds=bounds )
        self.p = res.x
        return res
    
    def train_diffusion(self,data_diffusion=[],y_key_diffsion="D"):
        dd=self.dd       
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion            
        bounds = [[0.01,1]]
        ferr = lambda x: self.error_diffusion(x,data_diffusion,y_key_diffsion)
        res = minimize(ferr, dd, bounds=bounds )
        self.dd = res.x
        return res

    def train_both(self,data=[],y_key="viscosity",
              data_diffusion=[],y_key_diffsion="D"):
        pdd=np.concat([self.p ,self.dd])
        if len(data)==0:
            data=self.data  
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion                         
        bounds = [[0.1,40],[-1.5,1.5],[0.1,8],[-1000,800],[-1000,800],[0.01,1]]
        ferr = lambda x: self.error_both(x,data,y_key)
        res = minimize(ferr, pdd, bounds=bounds )
        self.p = res.x[:-1]
        self.dd = [res.x[-1]]
        return res



class EyringEntropyModel_EES1():
    """
    ...
    """
    def __init__(self, parameters, data=[], p=[ 2,2, 1.4, .1, .1]):
        self.p = p
        self.data = data
        self.parameters = parameters
        return

    def predict_log_vf13(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        _,nf,_,_,_ = p
        n13 = 84446884.95844965
        vf13_0 =  data["V"]**(1/3) / n13
        return np.log(vf13_0) +  data["H_res"]/8.314/data["temperature"]*nf/3
        #return np.log(vf13_0) +  data["E_res"]/8.314/data["temperature"]/nf/3
        #return np.log(vf13_0) +  data["s_res"]/8.314/nf/3

    def predict_log_lambdas(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_,_ = p
        #sigma = self.parameters.pure_records[0].model_record.sigma/1e10
        #lambdas = 1/ (data["V"]-2*sigma**3*6.02214076e+23)
        lambdas = 1/ data["V"]
        return np.log(lambdas) - np.log(a**2)

    def predict_eyring_ref(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_,_ = p

        const = np.log(7.227814978226432) # np.sqrt(2*np.pi*8.31446261815324)
        c0 = np.log( np.sqrt(data["M"]*data["temperature"]) )

        log_vf13 = self.predict_log_vf13(p=p,data=data)
        log_lambdas = self.predict_log_lambdas(p=p,data=data)

        return const+c0+log_vf13+log_lambdas


    def predict_energy_barrier(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        _,_,n0,n1,n2 = p
        x = -data["s_res"]/8.314
        #x = -data["E_res"]/8.314/data["temperature"]
        #x = -data["H_res"]/8.314/data["temperature"]
        return n0*x + n1*x**2 + n2*x**3
    
    def predict(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        ey_ref = self.predict_eyring_ref(p=p,data=data)
        barrier = self.predict_energy_barrier(p=p,data=data)
        return ey_ref + barrier

    def predict_diffusion(self,p=[],data=[],dd=[1/6]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        log_dd = np.log(dd[0])
        log_vis = self.predict(p=p,data=data)
        n13 = 84446884.95844965
        log_vf13 =  np.log(data["V"]**(1/3) / n13)
        log_kT = np.log( 1.380649e-23*data["temperature"] )
        return log_kT -log_vis -log_vf13 + log_dd
    
    def error(self,p=[],data=[],y_key="viscosity"):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data            
        pred = self.predict(p=p,data=data)
        return np.mean( ( pred - np.log(data[y_key]) )**2 ) *1e10

    def train(self,p=[],data=[],y_key="viscosity"):
        if len(p)==0:
            p=self.p       
        if len(data)==0:
            data=self.data            
        bounds = [[0.1,40],[-0.5,0.5],[0.1,8],[-1000,800],[-1000,800]]
        ferr = lambda x: self.error(x,data,y_key)
        res = minimize(ferr, p, bounds=bounds )
        self.p = res.x
        return res

class EyringEntropyModel_EES0():
    """
    ...
    """
    def __init__(self, parameters, data=[], p=[ 2, 1.4, 3]):
        self.p = p
        self.data = data
        self.parameters = parameters
        return

    def predict_log_v13(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        b,_,_,_ = p

    def predict_log_vf13(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_ = p
        n13 = 84446884.95844965
        b = 2
        vf13_0 =  b*8.314*data["temperature"]*data["V"]**(1/3) / n13
        return np.log(vf13_0) -np.log( - data["temperature"] * data["s_res"] )

    def predict_log_lambdas(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_ = p
        lambdas = 1/ (data["V"])
        return np.log(lambdas) - np.log(a**2)        

    def predict_eyring_ref(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_ = p

        const = np.log(7.227814978226432) # np.sqrt(2*np.pi*8.31446261815324)
        c0 = np.log( np.sqrt(data["M"]*data["temperature"]) )

        log_vf13 = self.predict_log_vf13(p=p,data=data)
        log_lambdas = self.predict_log_lambdas(p=p,data=data)

        return const+c0+log_vf13+log_lambdas

    def predict_energy_barrier(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        _,n0,n1,n2 = p
        x = -data["s_res"]/8.314
        return n0*x + n1*x**2 + n2*x**3
    
    def predict(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        ey_ref = self.predict_eyring_ref(p=p,data=data)
        barrier = self.predict_energy_barrier(p=p,data=data)
        return ey_ref + barrier

    def error(self,p=[],data=[],y_key="viscosity"):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data            
        pred = self.predict(p=p,data=data)
        return np.mean( ( pred - np.log(data[y_key]) )**2 )*1e10

    def train(self,p=[],data=[],y_key="viscosity"):
        if len(p)==0:
            p=self.p       
        if len(data)==0:
            data=self.data            
        bounds = [[0.1,40],[0.1,8],[-1000,800],[-1000,800]]
        ferr = lambda x: self.error(x,data,y_key)
        res = minimize(ferr, p, bounds=bounds )
        self.p = res.x
        return res


def collision_integral( T, p):
    """
    computes analytical solution of the collision integral

    T: reduced temperature
    p: parameters

    returns analytical solution of the collision integral
    """
    A,B,C,D,E,F,G,H,R,S,W,P = p
    return A/T**B + C/np.exp(D*T) + E/np.exp(F*T) + G/np.exp(H*T) + R*T**B*np.sin(S*T**W - P)

def get_omega11(red_temperature):
    """
    computes analytical solution of the omega11 collision integral
    
    red_temperature: reduced temperature
    
    returns omega11
    """
    p11 = [ 
        1.06036,0.15610,0.19300,
        0.47635,1.03587,1.52996,
        1.76474,3.89411,0.0,
        0.0,0.0,0.0
    ]
    return collision_integral(red_temperature,p11)

def get_omega22(red_temperature):
    """
    computes analytical solution of the omega22 collision integral

    red_temperature: reduced temperature

    returns omega22
    """
    p22 = [ 
         1.16145,0.14874,0.52487,
         0.77320,2.16178,2.43787,
         0.0,0.0,-6.435/10**4,
         18.0323,-0.76830,7.27371
        ]
    return collision_integral(red_temperature,p22)

def get_viscosity_CE(temperature, saft_parameters):
    """
    computes viscosity reference for an array of temperatures
    uses pc-saft parameters

    temperature: array of temperatures
    saft_parameters: pc saft parameter object build with feos

    returns reference
    """
    epsilon = saft_parameters.pure_records[0].model_record.epsilon_k*KELVIN
    sigma   = saft_parameters.pure_records[0].model_record.sigma*ANGSTROM
    m       = saft_parameters.pure_records[0].model_record.m
    M       = saft_parameters.pure_records[0].molarweight*GRAM/MOL
    red_temperature = temperature/epsilon

    omega22 = get_omega22(red_temperature)

    sigma2 = sigma**2
    M_SI = M

    sq1  = np.sqrt( M_SI * KB * temperature / NAV /np.pi) # /METER**2 / KILOGRAM**2 *SECOND**2 ) *METER*KILOGRAM/SECOND
    div1 = omega22 * sigma2
    viscosity_reference = 5/16* sq1 / div1 #*PASCAL*SECOND
    viscosity_reference_m = 5/16* sq1 / div1/ m #*PASCAL*SECOND
    viscosity_reference_ig = 5/16* sq1 / sigma2
    return viscosity_reference, viscosity_reference_ig, viscosity_reference_m, omega22



def calc_stuff(sd, parameters):
    #print(sd)
    try:

        J_mol = JOULE/MOL
        J_molK = J_mol/KELVIN
        KG_m3 = KILO*GRAM/METER**3
        MOL_m3 = MOL/METER**3
        PS = PASCAL*SECOND
        
        eos = EquationOfState.pcsaft(parameters)
        M = parameters.pure_records[0].molarweight *(GRAM/MOL)
        m = parameters.pure_records[0].model_record.m
        epsilon = parameters.pure_records[0].model_record.epsilon_k*KELVIN
        sigma   = parameters.pure_records[0].model_record.sigma*ANGSTROM
        
        #sd = {"temperature":325*KELVIN, "pressure":2*BAR}
        if "pressure" in sd.keys():
            if sd["state"] == "L":
                state = State(eos, temperature=sd["temperature"]*KELVIN, pressure=sd["pressure"]*PASCAL, density_initialization="liquid")
                #print("liq")
            if sd["state"] == "G":
                state = State(eos, temperature=sd["temperature"]*KELVIN, pressure=sd["pressure"]*PASCAL, density_initialization="vapor")
                #print("vap")
            else:
                state = State(eos, temperature=sd["temperature"]*KELVIN, pressure=sd["pressure"]*PASCAL)  
            sd["rho"] = state.partial_density[0] / MOL_m3
        else:
            state = State(eos, temperature=sd["temperature"]*KELVIN, density=sd["rho"]*(MOL/METER**3) )
            sd["pressure"] = state.pressure() / PASCAL

        sd["s_total"] = state.specific_entropy(Contributions.Total) *M / J_molK
        sd["s_res"] = state.specific_entropy(Contributions.ResidualNvt) *M / J_molK
        sd["s_res*"] = sd["s_res"]/ KB /NAV
        sd["s_res**"] = sd["s_res*"]/m
        
        sd["E_res"] = state.specific_internal_energy(Contributions.ResidualNvt)*M / J_mol
        sd["H_res"] = state.specific_enthalpy(Contributions.ResidualNvt)*M / J_mol
        sd["G_res"] = state.specific_gibbs_energy(Contributions.ResidualNvt)*M / J_mol
        sd["A_res"] = state.specific_helmholtz_energy(Contributions.ResidualNvt)*M / J_mol
        sd["V"] = 1/sd["rho"]
        
        
        vle = PhaseDiagram.pure(eos,min_temperature=sd["temperature"]*KELVIN,npoints=100)
        vle_state_vapor = State(eos, temperature=vle.vapor.temperature[0], density=vle.vapor.density[0])
        vle_state_liquid = State(eos, temperature=vle.liquid.temperature[0], density=vle.liquid.density[0])
        sd["s_res_vle_gas"] = vle_state_vapor.specific_entropy(Contributions.ResidualNvt)*M  / J_molK
        sd["s_res_vle_liq"] = vle_state_liquid.specific_entropy(Contributions.ResidualNvt)*M  / J_molK
        sd["s_vap"] = vle_state_liquid.specific_entropy(Contributions.ResidualNvt)*M - vle_state_vapor.specific_entropy(Contributions.ResidualNvt)*M  
        sd["s_vap"] = sd["s_vap"]/ J_molK
        
        sd["dE_vap"] = vle_state_liquid.specific_internal_energy(Contributions.ResidualNvt)*M - vle_state_vapor.specific_internal_energy(Contributions.ResidualNvt)*M 
        sd["dE_vap"] = sd["dE_vap"]/ J_mol
        
        sd["dH_vap"] = vle_state_liquid.specific_enthalpy(Contributions.ResidualNvt)*M - vle_state_vapor.specific_enthalpy(Contributions.ResidualNvt)*M 
        sd["dH_vap"] = sd["dH_vap"]/ J_mol
        
        sd["rho_vap_liq"] = vle_state_liquid.partial_density[0] / MOL_m3
        sd["V_vap_liq"] = 1/sd["rho_vap_liq"]
        
        sd["rho_vap_gas"] = vle_state_vapor.partial_density[0] / MOL_m3
        sd["V_vap_gas"] = 1/sd["rho_vap_gas"]
        sd["dV_vap"] = sd["V_vap_liq"] - sd["V_vap_gas"]
        sd["dpressure"] = (vle_state_liquid.pressure() - vle_state_vapor.pressure())/PASCAL
        dummy = get_viscosity_CE(sd["temperature"]*KELVIN,parameters)
        sd["eta_CE"] = dummy[0] / PS
        sd["eta_CE_ig"] = dummy[1] / PS
        sd["eta_CE_m"] = dummy[2] / PS
        sd["omega22"] = dummy[3]
        sd["M"] = M /(KILO*GRAM/MOL)
        sd["m"] = m
        sd["sigma"] = sigma/METER 
        sd["epsilon"] = epsilon/KELVIN
        sd["R"] = RGAS / (JOULE/MOL/KELVIN)
        sd["success"] = 1
    except:
        sd["success"] = 0
    return sd


def get_lamda(T,parameter):
    epsilon = parameters.pure_records[0].model_record.epsilon_k    
    sigma = parameters.pure_records[0].model_record.sigma

    omega22 = get_omega22(T/epsilon)
    llambda = np.sqrt( np.sqrt(2)*np.pi*omega22  )*sigma /1e10
    return llambda


def get_VN(V):
    return (V/6.02214076e+23)**(1/3)

def get_V_eqd(T,parameters,pp):
    llambda1 = get_lamda(T,parameters)
    return llambda1**3*6.02214076e+23*pp**(3/2)
