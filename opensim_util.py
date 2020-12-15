import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
import math


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
