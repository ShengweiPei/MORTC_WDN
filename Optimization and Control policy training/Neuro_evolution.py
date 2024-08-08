# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:11:47 2023

@author: sp825
"""

from platypus.algorithms import NSGAII
from platypus import Hypervolume
from platypus.core import Problem
from platypus.types import Real



import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
np.random.seed(1)
import math
np.random.seed(1) # random seeds


import copy
import random
import pyDOE as pydoe
import sys
import xlrd 

import epanet

from epanet import toolkit as tk

import xlsxwriter 

EPSILON = sys.float_info.epsilon

import time


# 
#-------------------------------------------------------------
ENpro = tk.createproject()
tk.open(ENpro, "Anytown_revised_1h.inp", "Anytown_revised_1h.rpt", "")




nnodes = tk.getcount(ENpro, tk.NODECOUNT)
nlinks = tk.getcount(ENpro, tk.LINKCOUNT)

pump_group = []
for i in range (nlinks):
    link_type = tk.getlinktype(ENpro, i+1)
    link_id = tk.getlinkid(ENpro, i+1)
    # print(i+1, link_id, link_type)
    if link_type == 2:
        pump_group.append(i+1)
  
"""  """     
demand_group = []
tank_group = []
MeanDemand_group = []
for i in range(nnodes): 
    node_type = tk.getnodetype(ENpro, i+1)
    node_id = tk.getnodeid(ENpro, i+1)
    # print(i+1, node_id, node_type)
    
    
    bd = tk.getnodevalue(ENpro, i+1, tk.BASEDEMAND)
    # print(i+1, node_id, node_type, bd)
    
    if node_type == 0 and bd != 0:
        demand_group.append(i+1)
        MeanDemand_group.append(bd)
    elif node_type == 2:
        tank_group.append(i+1)
tk.close(ENpro)
tk.deleteproject(ENpro)

print(pump_group)
print(tank_group)
print(demand_group)


Multipliers = [1.0, 1.0, 1.0,0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1]
# print(len(Multipliers))

def DEMAND_Observation(basedemand_scenario, period):
    Demand_observation_group = np.zeros(19)
    multiplier = Multipliers[period] # 
    for j in range (len(basedemand_scenario)):
        Demand_observation_group[j] = multiplier*basedemand_scenario[j]
    return Demand_observation_group


def tanklevel_standardization(Tanklevel_Observation):
    # Tank levels are supposed to be between 10 and 35m
    # which is limited by the tank size of Anytown
    standard_tanklevel = []
    for i in range (len(Tanklevel_Observation)):
        sd_level = (Tanklevel_Observation[i] - 10)/25
        standard_tanklevel.append(sd_level)
    return standard_tanklevel


# some functions of Evolution Strategies
# --------------------------------------------------------------------------------


def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p


def build_net(len_state, len_action):
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    
    s0, p0 = linear(len_state, 30)
    s1, p1 = linear(30, 30)
    s2, p2 = linear(30, len_action)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))


def relu_func(inX):
    return np.maximum(0, inX)
    
    
    
def get_action(params, x):
    x = x[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    
    # print(x)
    return np.argmax(x, axis=1)[0]      # for discrete action
    # return 0.5 * np.tanh(x)[0] + 0.5             # for continuous action
    
LEN_STATE = 3 # 3 tank levels
LEN_ACTION = 4 # pump number = 0, 1, 2, 3

# training
net_shapes, net_params = build_net(LEN_STATE, LEN_ACTION)    
    


def get_reward_(params):
    params = np.array(params)
    network_params = params_reshape(net_shapes, params)
    EP_r = 0
    
    Operational_cost = 0
    Resilience = 0
    Constraint_cost = 0
    
    Results_Operational_cost = []
    Results_Resilience = []
    
    Results_PressureConstraint = []
    Results_TanklevelConstraint = []
    Results_SwitchConstraint = []
    
    
    # for demand scenario i
    for i in range (100):
        scenario_num = i
        
        operational_cost, resilience, constraint_cost = EPANET_Interaction(scenario_num, network_params)
        
        
        Operational_cost = Operational_cost + operational_cost
        Resilience = Resilience + resilience
        Constraint_cost = Constraint_cost + np.sum(constraint_cost)
        
        """ Results_recording; Results_recording; Results_recording; """
        Results_Operational_cost.append(operational_cost)
        Results_Resilience.append(resilience)
        
        Results_PressureConstraint.append(constraint_cost[0])
        Results_TanklevelConstraint.append(constraint_cost[1])
        Results_SwitchConstraint.append(constraint_cost[2])
        
        
    # Simulation_Results = [Results_Operational_cost, Results_Resilience, Results_PressureConstraint, Results_TanklevelConstraint, Results_SwitchConstraint]
        
    Mean_Resilience = np.mean(Results_Resilience)
    Mean_Operational_cost = np.mean(Results_Operational_cost)
    Mean_HydraulicConstr = np.mean(Results_PressureConstraint)
    Mean_TankleveConstr = np.mean(Results_TanklevelConstraint)
    Mean_SwitchConstr = np.mean(Results_SwitchConstraint)
    return Mean_Resilience, Mean_Operational_cost, Mean_HydraulicConstr, Mean_TankleveConstr, Mean_SwitchConstr


    

def training_scenario_generation(scenario_number):
    
    path_train = "LHS sample results/LhsRandom_results_for_training.xlsx"
    workbook_train = xlrd.open_workbook(path_train)
    data_sheet_train = workbook_train.sheet_by_index(0) #通过索引获取该列数据
    rowNum_train = data_sheet_train.nrows
    colNum_train = data_sheet_train.ncols
    
    demand_scenario = data_sheet_train.col_values(scenario_number)
    
    return demand_scenario





def EPANET_Interaction(scenario_num, network_params):
    MRIlist = []
    Pumpstatus_list = []
    Operational_cost = 0
    HydraulicConstr = 0
    TankBalanceConstr = 0
    SwitchConstr = 0
    unitprice = 0.12
    switchconstraint = 4
    
    Tanklevel_Observation_initial = [10, 10, 10]
    
    Tanklevel_Observation = [10, 10, 10]
    Basedemand_scenario = training_scenario_generation(scenario_num)
    EP_LEN = 24
    
    for period in range (EP_LEN):
        
        Demand_obs_current = DEMAND_Observation(Basedemand_scenario, period)
        Standard_tanklevel = tanklevel_standardization(Tanklevel_Observation)
        Standard_tanklevel = np.array(Standard_tanklevel)
        
        pump_solution = get_action(network_params, Standard_tanklevel)
        
        step_operational_cost, step_resilience, hydraulicConstr, final_tank_level, pump_states = Hydraulic_Simulation(pump_solution, Demand_obs_current, Tanklevel_Observation)
        
        Tanklevel_Observation = final_tank_level
        
        
        MRIlist.append(step_resilience)
        Pumpstatus_list.append(pump_states)
        
        Operational_cost = Operational_cost + step_operational_cost * unitprice
        HydraulicConstr = HydraulicConstr + hydraulicConstr
    
    resilience = np.mean(MRIlist)
    
    
    Tanklevel_Observation_final = final_tank_level
    for i in range(len(Tanklevel_Observation_final)):
        TankBalanceConstr += max(0, Tanklevel_Observation_initial[i] - Tanklevel_Observation_final[i])
    
    Switchtimes = 0
    for i in range (23):
        if Pumpstatus_list[i][0] != Pumpstatus_list[i+1][0]:
            Switchtimes += 1
    SwitchConstr += max(0, Switchtimes - switchconstraint)
    
    Switchtimes = 0
    for i in range (23):
        if Pumpstatus_list[i][1] != Pumpstatus_list[i+1][1]:
            Switchtimes += 1
    SwitchConstr += max(0, Switchtimes - switchconstraint)
    
    Switchtimes = 0
    for i in range (23):
        if Pumpstatus_list[i][2] != Pumpstatus_list[i+1][2]:
            Switchtimes += 1
    SwitchConstr += max(0, Switchtimes - switchconstraint)
    
    
    constraint_cost = [HydraulicConstr, TankBalanceConstr, SwitchConstr]
    
    
    return Operational_cost, resilience, constraint_cost



        
def Hydraulic_Simulation(pump_solution, demand_obs_current, Tanklevel_Observation):
    gpm_convert_cmh = 0.227124707
    feet_convert_meter = 0.3048
    pre_convert_meter = 0.6894757 #1 psi = 0.6894757 m
    specific_weight_water = 9.807
    requiredpressure = 40
    requiredhead = requiredpressure * pre_convert_meter
    
    step_operational_cost = 0
    PumpStatus = []
    HydraulicConstr = 0
    
    
    Ph = tk.createproject()
    tk.open(Ph, "Anytown_revised_1h.inp", "Anytown_revised_1h.rpt", "")

    for i in range (len(demand_group)):
        tk.setnodevalue(Ph, demand_group[i], tk.BASEDEMAND, demand_obs_current[i])
        
    if pump_solution == 0:
        tk.setpatternvalue(Ph, 2, 1, 0)
        tk.setpatternvalue(Ph, 3, 1, 0)
        tk.setpatternvalue(Ph, 4, 1, 0)
    elif pump_solution == 1:
        tk.setpatternvalue(Ph, 2, 1, 0)
        tk.setpatternvalue(Ph, 3, 1, 0)
        tk.setpatternvalue(Ph, 4, 1, 1)
    elif pump_solution == 2:
        tk.setpatternvalue(Ph, 2, 1, 0)
        tk.setpatternvalue(Ph, 3, 1, 1)
        tk.setpatternvalue(Ph, 4, 1, 1)
    else:
        tk.setpatternvalue(Ph, 2, 1, 1)
        tk.setpatternvalue(Ph, 3, 1, 1)
        tk.setpatternvalue(Ph, 4, 1, 1)
        
    # settings of tanks
    for i in range (len(tank_group)):
        # print("Tanklevel_Observation", Tanklevel_Observation)
        tk.setnodevalue(Ph, tank_group[i], tk.TANKLEVEL, Tanklevel_Observation[i])
        
    tk.setdemandmodel(Ph, tk.PDA, 0, 40, 0.5)   
        
    tk.openH(Ph)
    tk.initH(Ph, tk.NOSAVE)
    while True:
        apsum = 0
        bsum = 0
        t = tk.runH(Ph)
        
        if t == 0:
            for i in tank_group:
                d = tk.getnodevalue(Ph, i, tk.HEAD)
                e = tk.getnodevalue(Ph, i, tk.ELEVATION)
                
        
        if t % 3600 == 0:
            if t > 0:
                PRESSURE_ = []
                for i in demand_group:
                    d = tk.getnodevalue(Ph, i, tk.DEMAND)
                    if d != 0:
                        p = tk.getnodevalue(Ph, i, tk.PRESSURE)
                        PRESSURE_.append(p)
                        
                        h = tk.getnodevalue(Ph, i, tk.HEAD)
                        he = tk.getnodevalue(Ph, i, tk.ELEVATION)
                    apsum += d*((h-he)* feet_convert_meter - requiredpressure* pre_convert_meter)* gpm_convert_cmh
                    bsum += d*(he* feet_convert_meter + requiredpressure* pre_convert_meter)* gpm_convert_cmh
                
                critical_nodes_pressure = np.min(PRESSURE_)
                HydraulicConstr += max(0, requiredpressure-critical_nodes_pressure)
                
                
                step_resilience = apsum/bsum
            
                PumpStatus.append(tk.getlinkvalue(Ph, pump_group[0], tk.STATUS))
                PumpStatus.append(tk.getlinkvalue(Ph, pump_group[1], tk.STATUS))
                PumpStatus.append(tk.getlinkvalue(Ph, pump_group[2], tk.STATUS))
                
        tstep = tk.nextH(Ph)
        for i in pump_group:
            step_operational_cost += tk.getlinkvalue(Ph, i, tk.ENERGY)*tstep/3600
        
        if tstep <= 0:
            final_tank_level = []
            for i in tank_group:
                d = tk.getnodevalue(Ph, i, tk.HEAD)
                e = tk.getnodevalue(Ph, i, tk.ELEVATION)
                if d - e <= 10:
                    final_tank_level.append(10)
                    # Check the violation during hydraulic simulation process
                    # ErrorConstr = ErrorConstr 
                else:
                    final_tank_level.append(d - e)
                
            break
    
    tk.closeH(Ph)
    tk.close(Ph)
    tk.deleteproject(Ph)    
    
    return step_operational_cost, step_resilience, HydraulicConstr, final_tank_level, PumpStatus


class Anytownoperation(Problem):
    def __init__(self):
           
        super(Anytownoperation, self).__init__(nvars = len(net_params), nobjs=2, nconstrs= 3, )
        
        for i in range(self.nvars):
            self.types[i] = Real(-5, 5)
        self.constraints[:] = "<=0"
        self.directions = [Problem.MAXIMIZE, Problem.MINIMIZE]
  
    def simulation(self, solution):
        
        network_params = solution.variables
        # 
        Resilience_mean, Operational_cost, HydraulicConstr, TankBalanceConstr, SwitchConstr = get_reward_(network_params)
        return Resilience_mean, Operational_cost, HydraulicConstr, TankBalanceConstr, SwitchConstr

    def evaluate(self, solution):

        Resilience_mean, Operational_cost, HydraulicConstr, TankBalanceConstr, SwitchConstr = self.simulation(solution)
        solution.objectives[:] = [Resilience_mean, Operational_cost]
        solution.constraints[:] = [HydraulicConstr, TankBalanceConstr, SwitchConstr]
                




def NSGA2optimize():
    problem = Anytownoperation()
    algorithm = NSGAII(problem)
    # popsize = 100, ngens = 100
    # popsize = 100, ngens = 100. The hyperparameters should be set before running the code.
    
    algorithm.run(10000)
    
    
    print("============= results recording ===============")
    
    workbook_obj = xlsxwriter.Workbook( "Anytown_NE_1h/Training process/" +'objectives.xlsx')
    worksheet_obj = workbook_obj.add_worksheet('sheet1')
    
    workbook_nvars = xlsxwriter.Workbook( "Anytown_NE_1h/Training process/" +'network_params.xlsx')
    worksheet_nvars = workbook_nvars.add_worksheet('sheet1')
    
    workbook_constr = xlsxwriter.Workbook( "Anytown_NE_1h/Training process/" +'constraints.xlsx')
    worksheet_constr = workbook_constr.add_worksheet('sheet1')
    
    k = 0
    for s in algorithm.result:
        
        Results_obj = []
        for i in range (Anytownoperation().nobjs):
            Results_obj.append(s.objectives[i])
        worksheet_obj.write_row(k, 0, Results_obj)
        
        Results_nvars = []
        for i in range (Anytownoperation().nvars):
            Results_nvars.append(s.variables[i])
        worksheet_nvars.write_column(0, k, Results_nvars)
        
        Results_constr = []
        for i in range(Anytownoperation().nconstrs):
            Results_constr.append(s.constraints[i])
        worksheet_constr.write_row(k, 0, Results_constr)
        
        k += 1
        
        
    workbook_obj.close() 
    workbook_nvars.close()
    workbook_constr.close()
    
    
    plt.scatter([s.objectives[1] for s in algorithm.result], \
                [s.objectives[0] for s in algorithm.result])
    plt.ylabel("MRI")
    plt.xlabel("Operational Cost/$")
    filename = 'Anytown ' + "training process"
    plt.savefig("Anytown_NE_1h/Training process/" + filename)
    plt.close()



###########################################################################################
starttime = time.time()


NSGA2optimize()

endtime = time.time()
print ('running time:' , endtime - starttime)
print ("Test over!")