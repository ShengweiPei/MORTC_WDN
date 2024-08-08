# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 22:45:15 2023

@author: sp825
"""

""" pump solutions will be acquired respectively for each testing water demand scenario """

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
starttime = time.time()


Multipliers = [1.0, 1.0, 1.0,0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1]
# print(len(Multipliers))

def DEMAND_Observation(basedemand_scenario, period):
    Demand_observation_group = np.zeros(19)
    multiplier = Multipliers[period] # 
    for j in range (len(basedemand_scenario)):
        Demand_observation_group[j] = multiplier*basedemand_scenario[j]
    return Demand_observation_group

def MainLOOP_NSGAII(demand_scenario, demand_scenario_number):
    # print("num", demand_scenario_number, demand_scenario)
    class Anytownoperation(Problem):
        
       
        def __init__(self):
               
            super(Anytownoperation, self).__init__(nvars = 24, nobjs=2, nconstrs= 3, )
            
            for i in range(self.nvars):
                self.types[i] = Real(0, 3.9999999)
            self.constraints[:] = "<=0"
            self.directions = [Problem.MAXIMIZE, Problem.MINIMIZE]
            
            #constraint settings.
            self.requiredpressure = 40     #35,40,45
            # self.demandcoefficient = 1.1     #0.9,1.0,1.1
            
            
            
            
            self.demand_scenario = demand_scenario
            # print("这里这里这里", demand_scenario)
            
            self.switchconstraint = 4
            self.unitprice = 0.12
            
            self.gpm_convert_cmh = 0.227124707
            self.feet_convert_meter = 0.3048
            self.pre_convert_meter = 0.6894757 #1 psi = 0.6894757 m
            self.specific_weight_water = 9.807
            self.requiredhead = self.requiredpressure * self.pre_convert_meter
            
            
            Ph = tk.createproject()
            tk.open(Ph, "Anytown_revised_1h.inp", "Anytown_revised_1h.rpt", "")
            
            self.nnodes = tk.getcount(Ph, tk.NODECOUNT)
            self.nlinks = tk.getcount(Ph, tk.LINKCOUNT)
            
        
            self.demand_group = []
            self.tank_group = []
            self.pump_group = []
            self.demandmeans = []
            
            
            for i in range (self.nnodes):
                node_type = tk.getnodetype(Ph, i+1)
                # node_id = tk.getnodeid(ENpro, i+1)
                # print(i+1, node_id, node_type)
                bd = tk.getnodevalue(Ph, i+1, tk.BASEDEMAND)
                
                if node_type == 0 and bd != 0:
                    self.demand_group.append(i+1)
                    self.demandmeans.append(bd)
                elif node_type == 2:
                    self.tank_group.append(i+1)
            
            
            for i in range (self.nlinks):
                link_type = tk.getlinktype(Ph, i+1)
                # link_id = tk.getlinkid(ENpro, i+1)
                # print(i+1, link_id, link_type)
                if link_type == 2:
                    self.pump_group.append(i+1)
            
            tk.close(Ph)
            tk.deleteproject(Ph)
            
            # self.demandnodes = len(self.demand_group)
            
            # for k in self.demand_group:
            #     Mean_value = et.ENgetnodevalue(k,et.EN_BASEDEMAND)[1]
            #     self.demandmeans.append(Mean_value)  
            
    
            
            # #Setting the nodal demand with uncertainty for all demand nodes 
            # for k in range(len(self.demand_group)):
            #     et.ENsetnodevalue(self.demand_group[k], et.EN_BASEDEMAND, self.demandcoefficient*self.demandmeans[k])
                    
        def simulation(self, solution):
            MRIlist = []
            Pumpstatus_list = []
            Operational_cost = 0
            HydraulicConstr = 0
            TankBalanceConstr = 0
            SwitchConstr = 0
            
            Tanklevel_Observation_initial = [10, 10, 10]
            Tanklevel_Observation = [10, 10, 10]
            for period in range (24):
                onoroff = math.floor(solution.variables[period])
                MRI, PumpStatus, EnergyCal, step_hydraulicconstr, final_tank_level = self.hydraulicsimulation(period, onoroff, Tanklevel_Observation)
                
                Tanklevel_Observation = final_tank_level
                
                MRIlist.append(MRI)
                Pumpstatus_list.append(PumpStatus)
                
                Operational_cost = Operational_cost + EnergyCal*self.unitprice
                HydraulicConstr = HydraulicConstr + step_hydraulicconstr
                
                
            # Resilience calculation
            Resilience_mean = np.mean(MRIlist)
            # MRIsum = 0
            # for i in range(24):
            #     MRIsum += MRIlist[i+1]
            # Resilience_mean = (MRIsum + 0.5*MRIlist[0] + 0.5*MRIlist[24])/24
            
            
            Tanklevel_Observation_final = final_tank_level
            #Check the constraints of vialance for tank volume balance
            for i in range(len(Tanklevel_Observation_final)):
                TankBalanceConstr += max(0, Tanklevel_Observation_initial[i] - Tanklevel_Observation_final[i])
            
            Switchtimes = 0
            for i in range (23):
                if Pumpstatus_list[i][0] != Pumpstatus_list[i+1][0]:
                    Switchtimes += 1
            SwitchConstr += max(0, Switchtimes - self.switchconstraint)
            
            Switchtimes = 0
            for i in range (23):
                if Pumpstatus_list[i][1] != Pumpstatus_list[i+1][1]:
                    Switchtimes += 1
            SwitchConstr += max(0, Switchtimes - self.switchconstraint)
            
            Switchtimes = 0
            for i in range (23):
                if Pumpstatus_list[i][2] != Pumpstatus_list[i+1][2]:
                    Switchtimes += 1
            SwitchConstr += max(0, Switchtimes - self.switchconstraint)
            
            return Resilience_mean, Operational_cost, HydraulicConstr, TankBalanceConstr, SwitchConstr
    
        def hydraulicsimulation(self, period_number, onoroff, Tanklevel_Observation):
            # MRI = []
            PumpStatus = []
            EnergyCal = 0
            HydraulicConstr = 0
            
            
            Ph = tk.createproject()
            tk.open(Ph, "Anytown_revised_1h.inp", "Anytown_revised_1h.rpt", "")
            
            period_demand_observation = DEMAND_Observation(self.demand_scenario, period_number)
            for i in range (len(self.demand_group)):
                tk.setnodevalue(Ph, self.demand_group[i], tk.BASEDEMAND, period_demand_observation[i])
            
            
            if onoroff == 0:
                tk.setpatternvalue(Ph, 2, 1, 0)
                tk.setpatternvalue(Ph, 3, 1, 0)
                tk.setpatternvalue(Ph, 4, 1, 0)
            elif onoroff == 1:
                tk.setpatternvalue(Ph, 2, 1, 0)
                tk.setpatternvalue(Ph, 3, 1, 0)
                tk.setpatternvalue(Ph, 4, 1, 1)
            elif onoroff == 2:
                tk.setpatternvalue(Ph, 2, 1, 0)
                tk.setpatternvalue(Ph, 3, 1, 1)
                tk.setpatternvalue(Ph, 4, 1, 1)
            else:
                tk.setpatternvalue(Ph, 2, 1, 1)
                tk.setpatternvalue(Ph, 3, 1, 1)
                tk.setpatternvalue(Ph, 4, 1, 1)
                
            for i in range (len(self.tank_group)):
                # print("Tanklevel_Observation", Tanklevel_Observation)
                tk.setnodevalue(Ph, self.tank_group[i], tk.TANKLEVEL, Tanklevel_Observation[i])
                
            tk.setdemandmodel(Ph, tk.PDA, 0, 40, 0.5)  
                    
            tk.openH(Ph)
            tk.initH(Ph, tk.NOSAVE)
            while True:
                apsum = 0
                bsum = 0
                t = tk.runH(Ph)
                
                if t == 0:
                    Tanklevel_Start = []
                    for i in self.tank_group:
                        d = tk.getnodevalue(Ph, i, tk.HEAD)
                        e = tk.getnodevalue(Ph, i, tk.ELEVATION)
                        Tanklevel_Start.append(d-e)
                
                if t % 3600 == 0:
                    if t > 0:
                        for i in self.demand_group:
                            d = tk.getnodevalue(Ph, i, tk.DEMAND)
                            if d != 0:
                                p = tk.getnodevalue(Ph, i, tk.PRESSURE)
                                HydraulicConstr += max(0, self.requiredpressure-p)
                                
                                h = tk.getnodevalue(Ph, i, tk.HEAD)
                                he = tk.getnodevalue(Ph, i, tk.ELEVATION)
                            apsum += d*((h-he)*self.feet_convert_meter - self.requiredpressure*self.pre_convert_meter)*self.gpm_convert_cmh
                            bsum += d*(he*self.feet_convert_meter + self.requiredpressure*self.pre_convert_meter)*self.gpm_convert_cmh
                            
                        MRI = apsum/bsum
                    
                        PumpStatus.append(tk.getlinkvalue(Ph, self.pump_group[0], tk.STATUS))
                        PumpStatus.append(tk.getlinkvalue(Ph, self.pump_group[1], tk.STATUS))
                        PumpStatus.append(tk.getlinkvalue(Ph, self.pump_group[2], tk.STATUS))
                        
                tstep = tk.nextH(Ph)
                for i in self.pump_group:
                    EnergyCal += tk.getlinkvalue(Ph, i, tk.ENERGY)*tstep/3600
                
                if tstep <= 0:
                    final_tank_level = []
                    for i in self.tank_group:
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
            
            return MRI, PumpStatus, EnergyCal, HydraulicConstr, final_tank_level
                    
            
        def evaluate(self, solution):
    
            Resilience_mean, Operational_cost, HydraulicConstr, TankBalanceConstr, SwitchConstr = self.simulation(solution)
            solution.objectives[:] = [Resilience_mean, Operational_cost]
            solution.constraints[:] = [HydraulicConstr, TankBalanceConstr, SwitchConstr]
            
            
            
            
            
    
    
    def NSGA2optimize(demand_scenario, scenario_num):
        
        Ph = tk.createproject()
        tk.open(Ph, "Anytown_revised_1h.inp", "Anytown_revised_1h.rpt", "")
        
        
        problem = Anytownoperation()
        algorithm = NSGAII(problem)
        # popsize = 200, ngens = 1000
        algorithm.run(100000)
        
        
        
        tk.close(Ph)
        tk.deleteproject(Ph)
        
        
        
        print("scenario_num", scenario_num, "results recording")
        
        workbook_ps = xlsxwriter.Workbook( "Anytown_NSGAII/" + str(scenario_num) +'.xlsx')
        worksheet_ps = workbook_ps.add_worksheet('sheet1')
        
       
        k = 0
        for s in algorithm.result:
            Results_line = []
            
            Results_line.append(k)
            Results_line.append(" ")
            
            for i in range (Anytownoperation().nobjs):
                Results_line.append(s.objectives[i])
            Results_line.append(" ")
            
            for i in range (Anytownoperation().nvars):
                Results_line.append(math.floor(s.variables[i]))
            Results_line.append(" ")
            
            for i in range(Anytownoperation().nconstrs):
                Results_line.append(s.constraints[i])
                
            worksheet_ps.write_row(k, 0, Results_line)
            k += 1
        workbook_ps.close() 
        
        
        plt.scatter([s.objectives[1] for s in algorithm.result], \
                    [s.objectives[0] for s in algorithm.result])
        plt.ylabel("MRI")
        plt.xlabel("Operational Cost/$")
        filename = 'Anytown ' + "scenario_num " + str(scenario_num)
        plt.savefig("Anytown_NSGAII/" + filename)
        plt.close()
    
    
    NSGA2optimize(demand_scenario, demand_scenario_number)



path = "LHS sample results/LhsRandom_results_for_testing.xlsx"
workbook = xlrd.open_workbook(path)
data_sheet = workbook.sheet_by_index(0) #
rowNum = data_sheet.nrows
colNum = data_sheet.ncols

#print(rowNum,colNum)    
cols = np.zeros((colNum,rowNum))

# pool = multiprocessing.Pool(10) # 
for mmm in range (0, colNum):
    # 
    demand_scenario_number = mmm
    demand_scenario =  data_sheet.col_values(demand_scenario_number)

    MainLOOP_NSGAII(demand_scenario, demand_scenario_number)
    # pool.apply_async(func=MainLOOP_NSGAII,args=(demand_scenario, demand_scenario_number))
    # print(demand_scenario_number, "running finished")
# pool.close()
# pool.join()



endtime = time.time()
print ('running time:' , endtime - starttime)
print ("Test over!")
