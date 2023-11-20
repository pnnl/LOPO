import numpy as np
from datetime import date
import math
file_name = "portfolioQP.in"
file=open(file_name, 'w')

n_dim_list = [250]
project_fixedpt_list = [0,1]
slack_mode_list = [0,1]
initial_lb_ub_P_list = [(1/1000,1000)]
lb_ub_P_list = [(1/2,2),(1/4,4),(1/8,8),(1/16,16),(1/32,32),(1/100,100)]
#ub_P_list = []
initial_steps_list = [0,1,2]
final_steps_list = [0,2]
n_opt_steps_list   = [5,10,15,20,25]#,30,40,50]
seed_list = [0,1,2]



index = 1
for n_dim in n_dim_list:
    for initial_steps in initial_steps_list:
        for final_steps in final_steps_list:
            for slack_mode in slack_mode_list:
                for project_fixedpt in project_fixedpt_list:
                    for lb_ub_P in lb_ub_P_list:
                        lb_P, ub_P = lb_ub_P
                        for initial_lb_ub_P in initial_lb_ub_P_list:
                            initial_lb_P, initial_ub_P = initial_lb_ub_P
                            for n_opt_steps in n_opt_steps_list:
                                for seed in seed_list:
                                    print(  index,
                                            initial_steps,
                                            slack_mode,
                                            project_fixedpt,
                                            initial_lb_P,
                                            initial_ub_P,
                                            lb_P,
                                            ub_P,
                                            n_opt_steps,
                                            final_steps,
                                            seed,
                                            sep=',',
                                            file=file     )

                                    index = index + 1

file.close()
print("Done")
