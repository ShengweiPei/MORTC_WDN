# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 23:51:28 2024

@author: sp825
"""


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

import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

plt.rc('font',family='Times New Roman', weight = 'bold', size = 12,) 

font2 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 12,
         }


# 
#-------------------------------------------------------------
ENpro = tk.createproject()
tk.open(ENpro, "Anytown_revised_1h.inp", "Anytown_revised_1h.rpt", "")




nnodes = tk.getcount(ENpro, tk.NODECOUNT)
nlinks = tk.getcount(ENpro, tk.LINKCOUNT)

Main_pipe_group = []
pump_group = []
for i in range (nlinks):
    link_type = tk.getlinktype(ENpro, i+1)
    link_id = tk.getlinkid(ENpro, i+1)
    # print(i+1, link_id, link_type)
    if link_type == 2:
        pump_group.append(i+1)
    else:
        if link_id == "1":
            Main_pipe_group.append(i+1)
        if link_id == "2":
            Main_pipe_group.append(i+1)
        if link_id == "3":
            Main_pipe_group.append(i+1)
            
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

# print(pump_group)
# print(tank_group)
# print(demand_group)
print("Main_pipe_group", Main_pipe_group)









Multipliers = [1.0, 1.0, 1.0,0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1]
# print(len(Multipliers))
print(Multipliers[14:19])

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
    






def DemandSatisfaction(scenario_num, burst_pipe_num, Total_loss):
    
    basedemand_group = MeanDemand_group
    
    required_demand = np.sum(basedemand_group) * (Multipliers[burst_pipe_num] + Multipliers[burst_pipe_num + 1] + Multipliers[burst_pipe_num + 2])
    
    demand_satisfaction = 1 - Total_loss/required_demand
    return demand_satisfaction 



def SORT_(AAA, BBB):
    sort_AAA = np.sort(AAA)
    sort_AAA = sort_AAA[::-1]
    # print(sort_AAA)

    Sort_Index = np.argsort(AAA)


    sort_BBB = []
    for m in range (len(Sort_Index)):
        sort_index = Sort_Index[m]
        bbb = BBB[sort_index]
        sort_BBB.append(bbb)
    sort_BBB = sort_BBB[::-1]
        
    # print(sort_BBB) 
    return sort_AAA, sort_BBB


def get_distance_from_point_to_line(point, line_point1, line_point2):

    if line_point1 == line_point2:
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )

    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]

    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance


def Pumpsolution_selection(scenario_num, solution_num):
    path_NN = "E:/MORTC/Anytown/Anytown_NSGAII_Robust/Results_on_default scenario robust/" + str(scenario_num) + ".xlsx"
    workbook_NN = xlrd.open_workbook(path_NN)
    data_sheet_NN = workbook_NN.sheet_by_index(0) 
    rowNum_NN = data_sheet_NN.nrows
    colNum_NN = data_sheet_NN.ncols

    # reference_resilience = data_sheet_NN.col_values(2)
    # reference_cost = data_sheet_NN.col_values(3)
    
    normal_resilience = data_sheet_NN.row_values(solution_num)[1]
    
    path_NN = "E:/MORTC/Anytown/Anytown_NSGAII_Robust/universal_solutions.xlsx"
    workbook_NN = xlrd.open_workbook(path_NN)
    data_sheet_NN = workbook_NN.sheet_by_index(0) 
    rowNum_NN = data_sheet_NN.nrows
    colNum_NN = data_sheet_NN.ncols
    pump_solutions = data_sheet_NN.col_values(solution_num)
    
    return pump_solutions, normal_resilience


def EPANET_Interaction(burst_pipe_num, scenario_num, kkk, solution_num):
   
    
    selected_pump_solutions, normal_resilience = Pumpsolution_selection(scenario_num, solution_num)

    MRIlist = []
    Pumpstatus_list = []
    Operational_cost = 0
    HydraulicConstr = 0
    TankBalanceConstr = 0
    SwitchConstr = 0
    unitprice = 0.12
    switchconstraint = 4
    
    
    Sustain_time = 0
    Total_loss = 0
    
    Tanklevel_Observation_initial = [10, 10, 10]
    
    Tanklevel_Observation = [10, 10, 10]
    Basedemand_scenario = MeanDemand_group
    EP_LEN = 24
    
    for period in range (EP_LEN):
        
        Demand_obs_current = DEMAND_Observation(Basedemand_scenario, period)
        Standard_tanklevel = tanklevel_standardization(Tanklevel_Observation)
        Standard_tanklevel = np.array(Standard_tanklevel)
        
        
        # pump_solution = math.floor(selected_pump_solutions[period])
        pump_solution = selected_pump_solutions[period]
        
        step_operational_cost, step_resilience, hydraulicConstr, final_tank_level, pump_states, sustain_time, demand_loss = Hydraulic_Simulation(burst_pipe_num, pump_solution, Demand_obs_current, Tanklevel_Observation, period, kkk)
        
        Tanklevel_Observation = final_tank_level
        
        
        Sustain_time = Sustain_time + sustain_time
        Total_loss = Total_loss + demand_loss
        
        
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
    switch_times_1 = Switchtimes
    
    Switchtimes = 0
    for i in range (23):
        if Pumpstatus_list[i][1] != Pumpstatus_list[i+1][1]:
            Switchtimes += 1
    SwitchConstr += max(0, Switchtimes - switchconstraint)
    switch_times_2 = Switchtimes
    
    Switchtimes = 0
    for i in range (23):
        if Pumpstatus_list[i][2] != Pumpstatus_list[i+1][2]:
            Switchtimes += 1
    SwitchConstr += max(0, Switchtimes - switchconstraint)
    switch_times_3 = Switchtimes
    
    constraint_cost = [HydraulicConstr, TankBalanceConstr, SwitchConstr]
    switch_times = [switch_times_1, switch_times_2, switch_times_3]
    
    
    
    demand_satisfaction = DemandSatisfaction(scenario_num, burst_pipe_num, Total_loss)
    
    
    return Operational_cost, normal_resilience, constraint_cost, demand_satisfaction, selected_pump_solutions, switch_times



     
def Hydraulic_Simulation(burst_pipe_num, pump_solution, demand_obs_current, Tanklevel_Observation, period, kkk):
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
    

    if period == kkk or period == kkk+1 or period == kkk+2:
        tk.setlinkvalue(Ph, burst_pipe_num, tk.INITSTATUS, 0)
    
    # pipe_status = tk.getlinkvalue(Ph, burst_pipe_num, tk.INITSTATUS)
    # print("period", period, "pipe", burst_pipe_num, "status", pipe_status)
    
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
                
                
                if critical_nodes_pressure < requiredpressure:
                    # print("这里出现负压了！！！")
                    sustain_time = 3600
                    actual_demand = 0
                    for i in range (len(demand_group)):
                        demand_node = demand_group[i]
                        d = tk.getnodevalue(Ph, demand_node, tk.DEMAND)
                        actual_demand = actual_demand + d
                        
                        
                        demand_gap = demand_obs_current[i] - d
                        # if demand_gap < -0.001:
                        #     print("看看有没有小于0的情况发生", demand_gap)
                            
                        # if demand_gap <= 0:
                        #     print(" demand loss calculation is wrong !!!", demand_node, demand_gap)
                            
                    demand_loss = np.sum(demand_obs_current) - actual_demand
                        
                else:
                    sustain_time = 0
                    demand_loss = 0
                
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
    
    return step_operational_cost, step_resilience, HydraulicConstr, final_tank_level, PumpStatus, sustain_time, demand_loss








def Performance_under_pipe_bursts():
    
    # default water demand scenario
    scenario_number = -1
    

    for sm in range (200):
        
        workbook_res = xlsxwriter.Workbook( "E:/MORTC/Anytown_pipe_bursts/RO/ " + 'RO_under pipe bursts '+ str(sm) +'.xlsx')
        worksheet_res = workbook_res.add_worksheet('sheet1')
        
        solution_num = sm
        
        # print("进行到 solution_number", str(sm))
        
        fff = 0
        for m in range (3):
            burst_pipe_num = Main_pipe_group[m]
            
            Burst_time_period = np.arange(22)
            for kkk in range (len(Burst_time_period)):
                
                Operational_cost, normal_resilience, constraint_cost, demand_satisfaction, pump_solutions, switch_times = EPANET_Interaction(burst_pipe_num, scenario_number, kkk, solution_num)
                
                worksheet_res.write_row(fff, 0, [normal_resilience])
                worksheet_res.write_row(fff, 1, [Operational_cost])
                
                worksheet_res.write_row(fff, 4, [demand_satisfaction])
                worksheet_res.write_row(fff, 6, constraint_cost)
                
                worksheet_res.write_row(fff, 10, switch_times)
                worksheet_res.write_row(fff, 14, pump_solutions)
                
                fff = fff + 1
        workbook_res.close()
    




###########################################################################################
starttime = time.time()



Performance_under_pipe_bursts()


endtime = time.time()
print ('running time:' , endtime - starttime)
print ("Test over!")



