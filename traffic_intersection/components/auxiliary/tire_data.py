# Physical Data for Car Tyres
# Tung M. Phan
# California Institute of Technology
# August 2, 2018

def get_tire_data(designation):
    data = dict()
    if designation == '155SRS13':
        data['T_w'] = 6
        data['T_p'] = 24
        data['F_ZT'] = 810
        data['C_1'] = 1.0
        data['C_2'] = 0.34
        data['C_3'] = 0.57
        data['C_4'] = 0.32
        data['A_0'] = 914.02
        data['A_1'] = 12.9
        data['A_2'] = 2028.24
        data['A_3'] = 1.19
        data['A_4'] = -1019.2
        data['K_a'] = 0.05
        data['K_1'] = -0.0000122
        data['CS_FZ'] = 18.7
        data['mu_o'] = 0.85
    elif designation == 'P155/80D13':
        data['T_w'] = 6
        data['T_p'] = 24
        data['F_ZT'] = 900
        data['C_1'] = 0.535
        data['C_2'] = 1.05
        data['C_3'] = 1.15
        data['C_4'] = 0.8
        data['A_0'] = 1817
        data['A_1'] = 7.48
        data['A_2'] = 2455
        data['A_3'] = 1.857
        data['A_4'] = 3643
        data['K_a'] = 0.05
        data['K_1'] = -0.0000122
        data['CS_FZ'] = 18.7
        data['mu_o'] = 0.85
    elif designation == 'P185/70R13':
        data['T_w'] = 7.3
        data['T_p'] = 24
        data['F_ZT'] = 980
        data['C_1'] = 1.0
        data['C_2'] = 0.34
        data['C_3'] = 0.57
        data['C_4'] = 0.32
        data['A_0'] = 1068
        data['A_1'] = 11.3
        data['A_2'] = 2442.73
        data['A_3'] = 0.31
        data['A_4'] = -1877
        data['K_a'] = 0.05
        data['K_1'] = -0.000008
        data['CS_FZ'] = 17.91
        data['mu_o'] = 0.85
    return data
