import optuna
from get_mean import get_mean
from get_rg import get_rg
import subprocess
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging 
import sys 

def new_parameters(trial):

    # Obtain the new parameters from optuna and updated them in the parameters_search.dat file
    bond_t1_t2_k = trial.suggest_float("bond_t1_t2_k", 1.0, 12.0,step=0.001)
    bond_t2_t3_k = trial.suggest_float("bond_t2_t3_k", 1.0, 12.0,step=0.001)
    bond_t1_t3_k = trial.suggest_float("bond_t1_t3_k", 1.0, 12.0,step=0.001)
    bond_t3_t4_k = trial.suggest_float("bond_t3_t4_k", 1.0, 12.0,step=0.001)
    bond_t4_t5_k = trial.suggest_float("bond_t4_t5_k", 1.0, 12.0,step=0.001)
    bond_t5_t5_k = trial.suggest_float("bond_t5_t5_k", 1.0, 12.0,step=0.001)

    angle_t1_t2_t3_k = trial.suggest_float("angle_t1_t2_t3_k", 1.0, 12.0,step=0.001)
    angle_t1_t3_t2_k = trial.suggest_float("angle_t1_t3_t2_k", 1.0, 12.0,step=0.001)
    angle_t2_t1_t3_k = trial.suggest_float("angle_t2_t1_t3_k", 1.0, 12.0,step=0.001)
    angle_t2_t3_t4_k = trial.suggest_float("angle_t2_t3_t4_k", 1.0, 12.0,step=0.001)
    angle_t3_t4_t5_k = trial.suggest_float("angle_t3_t4_t5_k", 1.0, 12.0,step=0.001)
    angle_t4_t5_t5_k = trial.suggest_float("angle_t4_t5_t5_k", 1.0, 12.0,step=0.001)
    angle_t5_t5_t5_k = trial.suggest_float("angle_t5_t5_t5_k", 1.0, 12.0,step=0.001)

    angle_t1_t2_t3_theta = trial.suggest_float("angle_t1_t2_t3_theta", 70.0, 180.0,step=0.001)
    angle_t1_t3_t2_theta = trial.suggest_float("angle_t1_t3_t2_theta", 70.0, 180.0,step=0.001)
    angle_t2_t1_t3_theta = trial.suggest_float("angle_t2_t1_t3_theta", 70.0, 180.0,step=0.001)
    angle_t2_t3_t4_theta = trial.suggest_float("angle_t2_t3_t4_theta", 70.0, 180.0,step=0.001)
    angle_t3_t4_t5_theta = trial.suggest_float("angle_t3_t4_t5_theta", 70.0, 180.0,step=0.001)
    angle_t4_t5_t5_theta = trial.suggest_float("angle_t4_t5_t5_theta", 70.0, 180.0,step=0.001)
    angle_t5_t5_t5_theta = trial.suggest_float("angle_t5_t5_t5_theta", 70.0, 180.0,step=0.001) 

    sigma_t1 = trial.suggest_float("sigma_t1", 2.5, 5.0,step=0.001)
    sigma_t2 = trial.suggest_float("sigma_t2", 2.5, 5.0,step=0.001)
    sigma_t3 = trial.suggest_float("sigma_t3", 2.5, 5.0,step=0.001)
    sigma_t4 = trial.suggest_float("sigma_t4", 2.5, 5.0,step=0.001)
    sigma_t5 = trial.suggest_float("sigma_t5", 2.5, 5.0,step=0.001)

    epsilon_t1 = trial.suggest_float("epsilon_t1", 0.3, 1.5,step=0.001)
    epsilon_t2 = trial.suggest_float("epsilon_t2", 0.3, 1.5,step=0.001)
    epsilon_t3 = trial.suggest_float("epsilon_t3", 0.3, 1.5,step=0.001)
    epsilon_t4 = trial.suggest_float("epsilon_t4", 0.3, 1.5,step=0.001)
    epsilon_t5 = trial.suggest_float("epsilon_t5", 0.3, 1.5,step=0.001)

    gamma_r_t1 = trial.suggest_float("gamma_r_t1", 8, 16,step=0.001)
    gamma_r_t2 = trial.suggest_float("gamma_r_t2", 8, 16,step=0.001)
    gamma_r_t3 = trial.suggest_float("gamma_r_t3", 8, 16,step=0.001)
    gamma_r_t4 = trial.suggest_float("gamma_r_t4", 8, 16,step=0.001)
    gamma_r_t5 = trial.suggest_float("gamma_r_t5", 8, 16,step=0.001)

    l_t1t2 = trial.suggest_float("l_t1t2", 2.5, 5.0,step=0.001)
    l_t2t3 = trial.suggest_float("l_t2t3", 2.5, 5.0,step=0.001)
    l_t1t3 = trial.suggest_float("l_t1t3", 2.5, 5.0,step=0.001)
    l_t3t4 = trial.suggest_float("l_t3t4", 2.5, 5.0,step=0.001)
    l_t4t5 = trial.suggest_float("l_t4t5", 2.5, 5.0,step=0.001)
    l_t5t5 = trial.suggest_float("l_t5t5", 2.5, 5.0,step=0.001)

    subprocess.run(["cp", "parameters_search.dat", "parameters_search_aux.dat"], check=True)
    

    subprocess.run(["sed", "-i", f"s/k_bond_t1t2/{bond_t1_t2_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/k_bond_t2t3/{bond_t2_t3_k}/g", "parameters_search_aux.dat"], check=True) 
    subprocess.run(["sed", "-i", f"s/k_bond_t1t3/{bond_t1_t3_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/k_bond_t3t4/{bond_t3_t4_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/k_bond_t4t5/{bond_t4_t5_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/k_bond_t5t5/{bond_t5_t5_k}/g", "parameters_search_aux.dat"], check=True)
    
    subprocess.run(["sed", "-i", f"s/kt1t2t3/{angle_t1_t2_t3_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/kt1t3t2/{angle_t1_t3_t2_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/kt2t1t3/{angle_t2_t1_t3_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/kt2t3t4/{angle_t2_t3_t4_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/kt3t4t5/{angle_t3_t4_t5_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/kt4t5t5/{angle_t4_t5_t5_k}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/kt5t5t5/{angle_t5_t5_t5_k}/g", "parameters_search_aux.dat"], check=True)
 
    subprocess.run(["sed", "-i", f"s/theta_t1t2t3/{angle_t1_t2_t3_theta}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/theta_t1t3t2/{angle_t1_t3_t2_theta}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/theta_t2t1t3/{angle_t2_t1_t3_theta}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/theta_t2t3t4/{angle_t2_t3_t4_theta}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/theta_t3t4t5/{angle_t3_t4_t5_theta}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/theta_t4t5t5/{angle_t4_t5_t5_theta}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/theta_t5t5t5/{angle_t5_t5_t5_theta}/g", "parameters_search_aux.dat"], check=True)

    subprocess.run(["sed", "-i", f"s/sigma_t1/{sigma_t1}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/sigma_t2/{sigma_t2}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/sigma_t3/{sigma_t3}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/sigma_t4/{sigma_t4}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/sigma_t5/{sigma_t5}/g", "parameters_search_aux.dat"], check=True)

    subprocess.run(["sed", "-i", f"s/epsilon_1/{epsilon_t1}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/epsilon_2/{epsilon_t2}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/epsilon_3/{epsilon_t3}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/epsilon_4/{epsilon_t4}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/epsilon_5/{epsilon_t5}/g", "parameters_search_aux.dat"], check=True)

    subprocess.run(["sed", "-i", f"s/gamma_r1/{gamma_r_t1}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/gamma_r2/{gamma_r_t2}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/gamma_r3/{gamma_r_t3}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/gamma_r4/{gamma_r_t4}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/gamma_r5/{gamma_r_t5}/g", "parameters_search_aux.dat"], check=True)

    subprocess.run(["sed", "-i", f"s/l_t1t2/{l_t1t2}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/l_t2t3/{l_t2t3}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/l_t1t3/{l_t1t3}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/l_t3t4/{l_t3t4}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/l_t4t5/{l_t4t5}/g", "parameters_search_aux.dat"], check=True)
    subprocess.run(["sed", "-i", f"s/l_t5t5/{l_t5t5}/g", "parameters_search_aux.dat"], check=True)

    

def objective(trial):
    # Generate the model.
    model = new_parameters(trial)
    
    subprocess.run(["mpirun", "-np", "6", "lmp", "-in", "input.dat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["mpirun", "-np", "1", "lmp", "-in", "input_vaccum.dat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    objective = 0

    # Density

    x_1 = [150,175,200,225]
    x_2 = [425,450,475,500]
    y_1 = []
    for i in range(28, 32):
        filename = str(2) + "." + str(i) + "_averages.txt"
        y_1.append(get_mean(filename))
    
    reg_1 = LinearRegression().fit(np.array(x_1).reshape(-1, 1), y_1)
    dense_density_trial_1 = [reg_1.intercept_ + reg_1.coef_[0] * i for i in np.arange(150, 225,0.01)]

    a1,b1 = reg_1.coef_[0],reg_1.intercept_ # for T_g

    y_2 = []
    for i in range(39, 43):
        filename = str(2) + "." + str(i) + "_averages.txt"
        y_2.append(get_mean(filename))
    
    reg_2 = LinearRegression().fit(np.array(x_2).reshape(-1, 1), y_2)
    dense_density_trial_2 = [reg_2.intercept_ + reg_2.coef_[0] * i for i in np.arange(425, 500,0.01)]

    dense_density_trial = dense_density_trial_1 + dense_density_trial_2

    a2,b2 = reg_2.coef_[0],reg_2.intercept_ # for T_g

    # fit target para obter vetor denso
    target_1 = []
    y_1 = [1.2344, 1.2288, 1.2187, 1.2115]
    reg_1 = LinearRegression().fit(np.array(x_1).reshape(-1, 1), y_1)
    dense_density_target_1 = [reg_1.intercept_ + reg_1.coef_[0] * i for i in np.arange(150, 225,0.01)]

    target_2 = []
    y_2 = [1.1297, 1.1164, 1.0961, 1.0793]
    reg_2 = LinearRegression().fit(np.array(x_2).reshape(-1, 1), y_2)
    dense_density_target_2 = [reg_2.intercept_ + reg_2.coef_[0] * i for i in np.arange(425, 500,0.01)]

    dense_density_target = dense_density_target_1 + dense_density_target_2

    relative_error_density = 0
    for i, j in zip(dense_density_trial, dense_density_target):
        relative_error_density += ((i - j)/j) ** 2

    # Adding the slope to the objective function
    relative_error_slope = len(np.arange(150, 225,0.01)) * (a1-reg_1.coef_[0])**2/reg_1.coef_[0]**2 + \
                           len(np.arange(425, 500,0.01)) * (a2-reg_2.coef_[0])**2/reg_2.coef_[0]**2

    # Radius of Gyration
    
    rg_atomistic_vaccum = [4.471050544000001,4.4700970479999995,4.492482670999999,4.2140638175,4.021081141,\
            4.3125920015000005,4.479405487,4.360231142999999,4.299888399,4.393746198500001,4.956827014499999,\
            4.9121778955,5.067681224999999,5.186014581999999,4.96857314]

    rg_cg_vaccum = get_rg()   

    relative_error_rg = 0
    for j, i in zip(rg_atomistic_vaccum, rg_cg_vaccum):
        relative_error_rg += ((i - j)/j) ** 2
    relative_error_rg =  2 * len(np.arange(150, 225,0.01))/len(rg_cg_vaccum) * relative_error_rg
   
    objective = relative_error_density + relative_error_slope + relative_error_rg

    print(f"Relative Error Density: {relative_error_density:.6}\n")
    print(f"Relative Error Slope:   {relative_error_slope:.6}\n")
    print(f"Relative Error Rg:      {relative_error_rg:.6}\n")

    return objective


if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "All_Bonded_plus_sigmas_3000_trials" 
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,storage=storage_name,direction="minimize",load_if_exists=True)
    study.optimize(objective, n_trials=3000)
    

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))	


