import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
import math

# each frame: [pelvis(3), joint_angle(8), joint_vel(8), muscle_force(22), fiber_length(22), grf_l(1), grf_r(1)]
state_idx = {'pelvis_speed': 0, 'pelvis_tilt': 1, 'pelvis_ty': 2, 'hip_flexion_r': 3, 'hip_adduction_r': 4, 'knee_angle_r': 5, 'ankle_angle_r': 6,
             'hip_flexion_l': 7, 'hip_adduction_l': 8, 'knee_angle_l': 9, 'ankle_angle_l': 10, 'hip_flexion_r_v': 11, 'hip_adduction_r_v': 12,
             'knee_angle_r_v': 13, 'ankle_angle_r_v': 14, 'hip_flexion_l_v': 15, 'hip_adduction_l_v': 16, 'knee_angle_l_v': 17, 'ankle_angle_l_v': 18,
             'abd_r_f': 19, 'add_r_f': 20, 'hamstrings_r_f': 21, 'bifemsh_r_f': 22, 'glut_max_r_f': 23, 'iliopsoas_r_f': 24, 'rect_fem_r_f': 25,
             'vasti_r_f': 26, 'gastroc_r_f': 27, 'soleus_r_f': 28, 'tib_ant_r_f': 29, 'abd_l_f': 30, 'add_l_f': 31, 'hamstrings_l_f': 32, 'bifemsh_l_f': 33,
             'glut_max_l_f': 34, 'iliopsoas_l_f': 35, 'rect_fem_l_f': 36, 'vasti_l_f': 37, 'gastroc_l_f': 38, 'soleus_l_f': 39, 'tib_ant_l_f': 40,
             'abd_r_l': 41, 'add_r_l': 42, 'hamstrings_r_l': 43, 'bifemsh_r_l': 44, 'glut_max_r_l': 45, 'iliopsoas_r_l': 46, 'rect_fem_r_l': 47,
             'vasti_r_l': 48, 'gastroc_r_l': 49, 'soleus_r_l': 50, 'tib_ant_r_l': 51, 'abd_l_l': 52, 'add_l_l': 53, 'hamstrings_l_l': 54, 'bifemsh_l_l': 55,
             'glut_max_l_l': 56, 'iliopsoas_l_l': 57, 'rect_fem_l_l': 58, 'vasti_l_l': 59, 'gastroc_l_l': 60, 'soleus_l_l': 61, 'tib_ant_l_l': 62,
             'ground_force_1_vy': 63, 'ground_force_2_vy': 64}


def reward_shaping(state_, ref_traj):
    pelvis_err, angle_err, angle_vel_err, force_err, fiber_err, grf_err = [], [], [], [], [], []
    w_angle, w_angle_vel, w_pelvis = 0.4, 0.3, 0.3
    w_force, w_fiber, w_grf = 0.3, 0.3, 0.4

    assert w_angle + w_angle_vel + w_pelvis == 1.0 and w_force + w_fiber + w_grf == 1.0

    pelvis_err.append(ref_traj[state_idx['pelvis_speed']] - state_['pelvis']['vel'][0])
    pelvis_err.append(ref_traj[state_idx['pelvis_tilt']] - state_['pelvis']['pitch'])
    pelvis_err.append(ref_traj[state_idx['pelvis_ty']] - state_['pelvis']['height'])

    angle_err.append(ref_traj[state_idx['hip_flexion_r']] - state_['r_leg']['joint']['hip'])
    angle_err.append(ref_traj[state_idx['hip_adduction_r']] - state_['r_leg']['joint']['hip_abd'])
    angle_err.append(ref_traj[state_idx['knee_angle_r']] - state_['r_leg']['joint']['knee'])
    angle_err.append(ref_traj[state_idx['ankle_angle_r']] - state_['r_leg']['joint']['ankle'])
    angle_err.append(ref_traj[state_idx['hip_flexion_l']] - state_['l_leg']['joint']['hip'])
    angle_err.append(ref_traj[state_idx['hip_adduction_l']] - state_['l_leg']['joint']['hip_abd'])
    angle_err.append(ref_traj[state_idx['knee_angle_l']] - state_['l_leg']['joint']['knee'])
    angle_err.append(ref_traj[state_idx['ankle_angle_l']] - state_['l_leg']['joint']['ankle'])

    angle_vel_err.append(ref_traj[state_idx['hip_flexion_r_v']] - state_['r_leg']['d_joint']['hip'])
    angle_vel_err.append(ref_traj[state_idx['hip_adduction_r_v']] - state_['r_leg']['d_joint']['hip_abd'])
    angle_vel_err.append(ref_traj[state_idx['knee_angle_r_v']] - state_['r_leg']['d_joint']['knee'])
    angle_vel_err.append(ref_traj[state_idx['ankle_angle_r_v']] - state_['r_leg']['d_joint']['ankle'])
    angle_vel_err.append(ref_traj[state_idx['hip_flexion_l_v']] - state_['l_leg']['d_joint']['hip'])
    angle_vel_err.append(ref_traj[state_idx['hip_adduction_l_v']] - state_['l_leg']['d_joint']['hip_abd'])
    angle_vel_err.append(ref_traj[state_idx['knee_angle_l_v']] - state_['l_leg']['d_joint']['knee'])
    angle_vel_err.append(ref_traj[state_idx['ankle_angle_l_v']] - state_['l_leg']['d_joint']['ankle'])

    force_err.append(ref_traj[state_idx['abd_r_f']] - state_['r_leg']['HAB']['f'])
    force_err.append(ref_traj[state_idx['add_r_f']] - state_['r_leg']['HAD']['f'])
    force_err.append(ref_traj[state_idx['hamstrings_r_f']] - state_['r_leg']['HAM']['f'])
    force_err.append(ref_traj[state_idx['bifemsh_r_f']] - state_['r_leg']['BFSH']['f'])
    force_err.append(ref_traj[state_idx['glut_max_r_f']] - state_['r_leg']['GLU']['f'])
    force_err.append(ref_traj[state_idx['iliopsoas_r_f']] - state_['r_leg']['HFL']['f'])
    force_err.append(ref_traj[state_idx['rect_fem_r_f']] - state_['r_leg']['RF']['f'])
    force_err.append(ref_traj[state_idx['vasti_r_f']] - state_['r_leg']['VAS']['f'])
    force_err.append(ref_traj[state_idx['gastroc_r_f']] - state_['r_leg']['GAS']['f'])
    force_err.append(ref_traj[state_idx['soleus_r_f']] - state_['r_leg']['SOL']['f'])
    force_err.append(ref_traj[state_idx['tib_ant_r_f']] - state_['r_leg']['TA']['f'])
    force_err.append(ref_traj[state_idx['abd_l_f']] - state_['l_leg']['HAB']['f'])
    force_err.append(ref_traj[state_idx['add_l_f']] - state_['l_leg']['HAD']['f'])
    force_err.append(ref_traj[state_idx['hamstrings_l_f']] - state_['l_leg']['HAM']['f'])
    force_err.append(ref_traj[state_idx['bifemsh_l_f']] - state_['l_leg']['BFSH']['f'])
    force_err.append(ref_traj[state_idx['glut_max_l_f']] - state_['l_leg']['GLU']['f'])
    force_err.append(ref_traj[state_idx['iliopsoas_l_f']] - state_['l_leg']['HFL']['f'])
    force_err.append(ref_traj[state_idx['rect_fem_l_f']] - state_['l_leg']['RF']['f'])
    force_err.append(ref_traj[state_idx['vasti_l_f']] - state_['l_leg']['VAS']['f'])
    force_err.append(ref_traj[state_idx['gastroc_l_f']] - state_['l_leg']['GAS']['f'])
    force_err.append(ref_traj[state_idx['soleus_l_f']] - state_['l_leg']['SOL']['f'])
    force_err.append(ref_traj[state_idx['tib_ant_l_f']] - state_['l_leg']['TA']['f'])

    fiber_err.append(ref_traj[state_idx['abd_r_l']] - state_['r_leg']['HAB']['l'])
    fiber_err.append(ref_traj[state_idx['add_r_l']] - state_['r_leg']['HAD']['l'])
    fiber_err.append(ref_traj[state_idx['hamstrings_r_l']] - state_['r_leg']['HAM']['l'])
    fiber_err.append(ref_traj[state_idx['bifemsh_r_l']] - state_['r_leg']['BFSH']['l'])
    fiber_err.append(ref_traj[state_idx['glut_max_r_l']] - state_['r_leg']['GLU']['l'])
    fiber_err.append(ref_traj[state_idx['iliopsoas_r_l']] - state_['r_leg']['HFL']['l'])
    fiber_err.append(ref_traj[state_idx['rect_fem_r_l']] - state_['r_leg']['RF']['l'])
    fiber_err.append(ref_traj[state_idx['vasti_r_l']] - state_['r_leg']['VAS']['l'])
    fiber_err.append(ref_traj[state_idx['gastroc_r_l']] - state_['r_leg']['GAS']['l'])
    fiber_err.append(ref_traj[state_idx['soleus_r_l']] - state_['r_leg']['SOL']['l'])
    fiber_err.append(ref_traj[state_idx['tib_ant_r_l']] - state_['r_leg']['TA']['l'])
    fiber_err.append(ref_traj[state_idx['abd_l_l']] - state_['l_leg']['HAB']['l'])
    fiber_err.append(ref_traj[state_idx['add_l_l']] - state_['l_leg']['HAD']['l'])
    fiber_err.append(ref_traj[state_idx['hamstrings_l_l']] - state_['l_leg']['HAM']['l'])
    fiber_err.append(ref_traj[state_idx['bifemsh_l_l']] - state_['l_leg']['BFSH']['l'])
    fiber_err.append(ref_traj[state_idx['glut_max_l_l']] - state_['l_leg']['GLU']['l'])
    fiber_err.append(ref_traj[state_idx['iliopsoas_l_l']] - state_['l_leg']['HFL']['l'])
    fiber_err.append(ref_traj[state_idx['rect_fem_l_l']] - state_['l_leg']['RF']['l'])
    fiber_err.append(ref_traj[state_idx['vasti_l_l']] - state_['l_leg']['VAS']['l'])
    fiber_err.append(ref_traj[state_idx['gastroc_l_l']] - state_['l_leg']['GAS']['l'])
    fiber_err.append(ref_traj[state_idx['soleus_l_l']] - state_['l_leg']['SOL']['l'])
    fiber_err.append(ref_traj[state_idx['tib_ant_l_l']] - state_['l_leg']['TA']['l'])

    grf_err.append(ref_traj[state_idx['ground_force_1_vy']] - state_['l_leg']['ground_reaction_forces'][2])
    grf_err.append(ref_traj[state_idx['ground_force_2_vy']] - state_['r_leg']['ground_reaction_forces'][2])

    r_kinematic, r_dynamic = 0, 0
    r_kinematic += w_angle * np.exp(-np.linalg.norm(angle_err))
    r_kinematic += w_angle_vel * np.exp(-np.linalg.norm(angle_vel_err))
    r_kinematic += w_pelvis * np.exp(-np.linalg.norm(pelvis_err))

    r_dynamic += w_force * np.exp(-np.linalg.norm(force_err))
    r_dynamic += w_fiber * np.exp(-np.linalg.norm(fiber_err))
    r_dynamic += w_grf * np.exp(-np.linalg.norm(grf_err))

    # print(np.linalg.norm(pelvis_err), np.linalg.norm(angle_err), np.linalg.norm(angle_vel_err),
    # np.linalg.norm(activation_err), np.linalg.norm(fiber_err), np.linalg.norm(grf_err), 'r=', r)
    return 0.5 * r_kinematic + 0.5 * r_dynamic


def lowpass_grf(original_file, fe, output_file):
    with open(original_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            for _ in range(7):
                line = fin.readline()
                fout.write(line)

            titles_data = fin.readline()
            fout.write(titles_data)

            data_list = fin.readlines()
            mat = None
            for line in data_list:
                row_list = line.split('\t')
                row = np.array([])
                for e in row_list:
                    row = np.append(row, eval(e))
                mat = row if mat is None else np.vstack((mat, row))

            wn = 2 * fe / 2000  # fe(Hz) lowpass filter
            b, a = signal.butter(N=4, Wn=wn, fs=200)
            for col in range(1, mat.shape[1]):
                mat[:, col] = signal.filtfilt(b, a, mat[:, col])

            for row in range(mat.shape[0]):
                for col in range(mat.shape[1]):
                    if col == mat.shape[1] - 1:
                        fout.write('%f\n' % mat[row][col])
                    else:
                        fout.write('%f\t' % mat[row][col])


def load_muscle_force(filename):
    with open(filename, 'r') as f:
        for _ in range(22):
            f.readline()
        titles = f.readline().split()
        title_idx = {}
        idx = 0
        for name in titles:
            title_idx[name] = idx
            idx += 1

        data_list = f.readlines()
        mat = None
        for line in data_list:
            row_list = line.split('\t')
            if 0.03 <= eval(row_list[0]) <= 1.41:
                row = np.array([])
                for e in row_list:
                    row = np.append(row, eval(e))
                mat = row if mat is None else np.vstack((mat, row))

        # fitting and interpolation
        new_mat = np.arange(0.03, 1.42, 0.01)
        for col in range(1, mat.shape[1]):
            f = interpolate.interp1d(mat[:, 0], mat[:, col], kind='cubic')
            x = list(np.arange(0.03, 1.41, 0.01))
            new_row = np.append(f(x), mat[-1, col])
            new_mat = np.vstack((new_mat, new_row))

        print('nRows:', new_mat.T.shape[0], '\tnColumns:', new_mat.T.shape[1])
        return new_mat.T, title_idx


def load_grf(filename):
    with open(filename, 'r') as f:
        for _ in range(7):
            f.readline()
        titles = f.readline().split()
        title_idx = {}
        idx = 0
        for name in titles:
            title_idx[name] = idx
            idx += 1

        data_list = f.readlines()
        mat = None
        for line in data_list:
            row_list = line.split('\t')
            if 0.03 <= eval(row_list[0]) <= 1.41:
                if math.modf(eval(row_list[0]) * 100)[0] <= 0.001 or math.modf(eval(row_list[0]) * 100)[0] >= 0.999:
                    row = np.array([])
                    for e in row_list:
                        row = np.append(row, eval(e))
                    mat = row if mat is None else np.vstack((mat, row))
        print('nRows:', mat.shape[0], '\tnColumns:', mat.shape[1])

        # title = 'ground_moment_2_mz'
        # idx = titles.index(title)
        # data = mat[:, idx]
        #
        # plt.plot(data)
        # plt.title(title)
        # plt.savefig('pic/' + title + '.png')
        # plt.show()

        return mat, title_idx


def load_cmc(filename):  # load gait14dof22musc_walk1_states.sto and subsample at 100Hz
    with open(filename, 'r') as f:
        for _ in range(6):
            f.readline()

        titles = f.readline().split()
        title_idx = {}
        idx = 0
        for name in titles:
            title_idx[name] = idx
            idx += 1

        data_list = f.readlines()
        mat = None
        for line in data_list:
            row_list = line.split('\t')
            if 0.03 <= eval(row_list[0]) <= 1.41:
                if math.modf(eval(row_list[0]) * 100)[0] <= 0.001 or math.modf(eval(row_list[0]) * 100)[0] >= 0.999:
                    row = []
                    for i in row_list:
                        row.append(eval(i))
                    mat = np.array(row) if mat is None else np.vstack((mat, np.array(row)))
        print('nRows:', mat.shape[0], '\tnColumns:', mat.shape[1])
        return mat, title_idx

def print_activation_cmc(filename, muscle_name):  # load gait14dof22musc_walk1_states.sto and print certain muscle's value
    with open(filename, 'r') as f:
        for _ in range(6):
            f.readline()

        titles = f.readline().split()
        idx = titles.index('/forceset/' + muscle_name + '/activation')

        data_list = f.readlines()
        res = []
        for line in data_list:
            row_list = line.split('\t')
            if 0.03 <= eval(row_list[0]) <= 1.41:
                if math.modf(eval(row_list[0]) * 100)[0] <= 0.001 or math.modf(eval(row_list[0]) * 100)[0] >= 0.999:
                    res.append(eval(row_list[idx]))
                    
        return res

def get_ref_traj(cmc_file, grf_file, muscle_file):
    grf_mat, grf_title_idx = load_grf(grf_file)
    cmc_mat, cmc_title_idx = load_cmc(cmc_file)
    force_mat, force_title_idx = load_muscle_force(muscle_file)
    assert grf_mat.shape[0] == cmc_mat.shape[0] == force_mat.shape[0]
    frame_n = grf_mat.shape[0]

    # each frame: [pelvis(3), joint_angle(8), joint_vel(8), muscle_force(22), fiber_length(22), grf_l, grf_r]
    pelvis = ['ground_pelvis/pelvis_tx/speed', 'ground_pelvis/pelvis_tilt/value', 'ground_pelvis/pelvis_ty/value']
    joints = ['hip_r/hip_adduction_r', 'hip_r/hip_flexion_r', 'knee_r/knee_angle_r', 'ankle_r/ankle_angle_r',
              'hip_l/hip_flexion_l', 'hip_l/hip_adduction_l', 'knee_l/knee_angle_l', 'ankle_l/ankle_angle_l']
    muscles = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r', 'rect_fem_r',
               'vasti_r', 'gastroc_r', 'soleus_r', 'tib_ant_r', 'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l',
               'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l', 'gastroc_l', 'soleus_l', 'tib_ant_l']
    grfs = ['ground_force_1_vy', 'ground_force_2_vy']

    traj_ref = None
    for row in range(frame_n):
        frame = []
        for pel in pelvis:
            frame.append(cmc_mat[row][cmc_title_idx['/jointset/' + pel]])
        for joint in joints:  # joint angle
            frame.append(cmc_mat[row][cmc_title_idx['/jointset/' + joint + '/value']])
        for joint in joints:  # joint velocity
            frame.append(cmc_mat[row][cmc_title_idx['/jointset/' + joint + '/speed']])
        for muscle in muscles:  # muscle force
            frame.append(force_mat[row][force_title_idx[muscle]])
        for muscle in muscles:  # muscle fiber length
            frame.append(cmc_mat[row][cmc_title_idx['/forceset/' + muscle + '/fiber_length']])
        for grf in grfs:  # ground reaction force
            frame.append(grf_mat[row][grf_title_idx[grf]])
        traj_ref = np.array(frame) if traj_ref is None else np.vstack((traj_ref, np.array(frame)))

    # normalization
    for col in range(traj_ref.shape[1]):
        traj_ref[:, col] = (traj_ref[:, col] - traj_ref[:, col].mean()) / traj_ref[:, col].std()
    np.save('reference_trajectory.npy', traj_ref)
    print('Successfully save reference trajectory!')


def main():
    # lowpass_grf('subject01_walk1_grf.mot', 600, 'subject01_walk1_grf_filtered.mot')  # 600Hz lowpass
    # get_ref_traj('gait14dof22musc_walk1_states.sto', 'subject01_walk1_grf_filtered.mot', 'gait14dof22musc_walk1_Actuation_force.sto')
    add_r = print_activation_cmc('gait14dof22musc_walk1_states.sto', 'add_r')

    plt.figure()
    plt.title('ADD')
    plt.xlabel('Timestep')
    plt.ylabel('Muscle Activation')
    plt.plot(add_r)
    plt.show()

if __name__ == '__main__':
    main()
