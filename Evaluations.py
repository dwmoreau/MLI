import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def evaluate_regression(data, n_outputs, unit_cell_key, save_to_name, y_indices):
    alpha = 0.1
    markersize = 0.5

    data = data[~data['augmented']]
    figsize = (n_outputs*2 + 2, 10)
    fig, axes = plt.subplots(5, n_outputs, figsize=figsize)
    if n_outputs == 1:
        axes = axes[:, np.newaxis]
    y_true = np.stack(data[unit_cell_key])[:, y_indices]
    y_pred = np.stack(data[f'{unit_cell_key}_pred'])
    y_cov = np.stack(data[f'{unit_cell_key}_pred_cov'])
    y_std = np.sqrt(np.diagonal(y_cov, axis1=1, axis2=2))
    y_error = np.abs(y_pred - y_true)
    titles = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']

    for uc_index in range(n_outputs):
        all_info = np.sort(np.concatenate((y_true[:, uc_index], y_pred[:, uc_index])))
        lower = all_info[int(0.005*all_info.size)]
        upper = all_info[int(0.995*all_info.size)]

        all_error = np.sort(np.concatenate((y_error[:, uc_index], y_std[:, uc_index])))
        lower_error = all_error[int(0.005*all_error.size)]
        upper_error = all_error[int(0.995*all_error.size)]

        if upper > lower:
            axes[0, uc_index].plot(
                y_true[:, uc_index], y_pred[:, uc_index],
                color=[0, 0, 0], alpha=alpha,
                linestyle='none', marker='.', markersize=markersize,
                )
            axes[0, uc_index].plot(
                [lower, upper], [lower, upper],
                color=[0.7, 0, 0], linestyle='dotted'
                )
            axes[1, uc_index].semilogy(
                y_true[:, uc_index], y_error[:, uc_index],
                color=[0, 0, 0], alpha=alpha,
                linestyle='none', marker='.', markersize=markersize,
                )
            axes[2, uc_index].semilogy(
                y_true[:, uc_index], y_std[:, uc_index],
                color=[0, 0, 0], alpha=alpha,
                linestyle='none', marker='.', markersize=markersize,
                )

        if upper_error > lower_error:
            bins = np.linspace(lower_error, upper_error, 101)
            centers = (bins[1:] + bins[:-1]) / 2
            dbin = centers[1] - centers[0]
            hist_std, _ = np.histogram(y_std[:, uc_index], bins=bins, density=True)
            hist_error, _ = np.histogram(y_error[:, uc_index], bins=bins, density=True)
            axes[3, uc_index].loglog(
                [lower_error, upper_error], [lower_error, upper_error],
                color=[0.7, 0, 0], linestyle='dotted'
                )
            axes[3, uc_index].loglog(
                y_error[:, uc_index], y_std[:, uc_index],
                color=[0, 0, 0], alpha=alpha,
                linestyle='none', marker='.', markersize=markersize,
                )
            axes[4, uc_index].bar(centers, hist_error, width=dbin, label='Error', alpha=0.5)
            axes[4, uc_index].bar(centers, hist_std, width=dbin, label='STD Est', alpha=0.5)

        if upper > lower:
            axes[0, uc_index].set_xlim([lower, upper])
            axes[0, uc_index].set_ylim([lower, upper])
            axes[1, uc_index].set_xlim([lower, upper])
            axes[2, uc_index].set_xlim([lower, upper])
        if upper_error > 0:
            axes[1, uc_index].set_ylim([lower_error, upper_error])
            axes[2, uc_index].set_ylim([lower_error, upper_error])
            axes[3, uc_index].set_xlim([lower_error, upper_error])
            axes[3, uc_index].set_ylim([lower_error, upper_error])
            axes[4, uc_index].set_xscale('log')

        error = np.sort(y_error[:, uc_index])
        p25 = error[int(0.25 * error.size)]
        p50 = error[int(0.50 * error.size)]
        p75 = error[int(0.75 * error.size)]
        rmse = np.sqrt(1/len(data) * np.linalg.norm(error)**2)
        error_titles = [
            titles[uc_index],
            f'RMSE: {rmse:0.4f}',
            f'25%: {p25:0.4f}',
            f'50%: {p50:0.4f}',
            f'75%: {p75:0.4f}',
            ]
        axes[0, uc_index].set_title('\n'.join(error_titles), fontsize=12)
        axes[0, uc_index].set_xlabel('True')
        axes[1, uc_index].set_xlabel('True')
        axes[2, uc_index].set_xlabel('True')
        axes[3, uc_index].set_xlabel('Error')
        axes[4, uc_index].set_xlabel('Error / STD')
    axes[0, 0].set_ylabel('Predicted')
    axes[1, 0].set_ylabel('Error')
    axes[2, 0].set_ylabel('STD Est')
    axes[3, 0].set_ylabel('STD Est')
    axes[4, 0].set_ylabel('Distribution')
    axes[4, 0].legend(frameon=False)
    fig.tight_layout()
    #fig.savefig()
    fig.savefig(save_to_name)
    plt.close()

def calibrate_regression(data, n_outputs, unit_cell_key, save_to_name, y_indices):
    # calculate residuals / uncertainty
    hist_bins = np.linspace(-4, 4, 101)
    hist_centers = (hist_bins[1:] + hist_bins[:-1]) / 2
    n_calib_bins = 25
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    alphas = [1, 0.5]
    labels = ['Training', 'Validation']

    data = data[~data['augmented']]
    data = [
        data[data['train']],
        data[~data['train']]
        ]
    figsize=(10, 2 * (n_outputs + 1))
    fig, axes = plt.subplots(n_outputs, 3, figsize=figsize)
    if n_outputs == 1:
        axes = axes[np.newaxis, :]
    cv = np.zeros((n_outputs, 2))
    ENCE = np.zeros((n_outputs, 2, n_calib_bins))
    for train_index in range(2):
        y_true = np.stack(data[train_index][unit_cell_key])[:, y_indices]
        y_pred = np.stack(data[train_index][f'{unit_cell_key}_pred'])
        y_cov = np.stack(data[train_index][f'{unit_cell_key}_pred_cov'])
        y_std = np.sqrt(np.diagonal(y_cov, axis1=1, axis2=2))
        y_error = y_pred - y_true

        for index in range(n_outputs):
            z = y_error[:, index] / y_std[:, index]
            hist, _ = np.histogram(z, bins=hist_bins, density=True)
            axes[index, 0].bar(
                hist_centers, hist, width=(hist_bins[1] - hist_bins[0]),
                color=colors[train_index], alpha=alphas[train_index], label=labels[train_index]
                )

            sorted_std = np.sort(y_std[:, index])
            lower_std = sorted_std[int(0.01*sorted_std.size)]
            upper_std = sorted_std[int(0.98*sorted_std.size)]
            bins_std = np.linspace(lower_std, upper_std, n_calib_bins + 1)
            centers_std = (bins_std[1:] + bins_std[:-1]) / 2
            y_error_binned = np.zeros((n_calib_bins, 2))
            for bin_index in range(n_calib_bins):
                indices = np.logical_and(
                    y_std[:, index] >= bins_std[bin_index],
                    y_std[:, index] < bins_std[bin_index + 1]
                    )
                if np.sum(indices) > 0:
                    y_error_binned[bin_index, 0] = np.abs(y_error[indices, index]).mean()
                    y_error_binned[bin_index, 1] = np.abs(y_error[indices, index]).std()
                    RMV = np.sqrt(np.mean(y_std[indices, index]**2))
                    RMSE = np.sqrt(np.mean(y_error[indices, index]**2))
                    ENCE[index, train_index, bin_index] = np.abs(RMV - RMSE) / RMV
            mean_sigma = y_std[:, index].mean()
            numerator = np.sum((y_std[:, index] - mean_sigma)**2) / (y_std[:, index].size - 1)
            cv[index, train_index] = np.sqrt(numerator) / mean_sigma

            axes[index, 1].errorbar(
                centers_std, y_error_binned[:, 0], y_error_binned[:, 1],
                marker='.', color=colors[train_index], label=labels[train_index]
                )

            sorted_error = np.sort(np.abs(y_error[:, index]))
            lower_error = sorted_error[int(0.01*sorted_error.size)]
            upper_error = sorted_error[int(0.98*sorted_error.size)]
            bins_error = np.linspace(lower_error, upper_error, n_calib_bins + 1)
            centers_error = (bins_error[1:] + bins_error[:-1]) / 2
            y_std_binned = np.zeros((n_calib_bins, 2))
            for bin_index in range(n_calib_bins):
                indices = np.logical_and(
                    y_error[:, index] >= bins_error[bin_index],
                    y_error[:, index] < bins_error[bin_index + 1]
                    )
                y_std_binned[bin_index, 0] = np.abs(y_std[indices, index]).mean()
                y_std_binned[bin_index, 1] = np.abs(y_std[indices, index]).std()

            axes[index, 2].errorbar(
                centers_error, y_std_binned[:, 0], y_std_binned[:, 1],
                marker='.', color=colors[train_index], label=labels[train_index]
                )

    for index in range(n_outputs):
        axes[index, 1].annotate(
            '\n'.join((
                f'ENCE: {np.mean(ENCE[index, 0]):0.3f} / {np.mean(ENCE[index, 1]):0.3f}',
                f'Cv: {cv[index, 0]:0.3f} / {cv[index, 1]:0.3f}'
                )),
            xy=(0.05, 0.85), xycoords='axes fraction'
            )
        axes[index, 0].plot(
            hist_centers,
            scipy.stats.norm.pdf(hist_centers),
            color=[0, 0, 0],
            linestyle='dotted'
            )
        axes[index, 2].set_ylabel('Mean STD Estimate')
        axes[index, 0].set_ylabel('Distribution')
        axes[index, 1].set_ylabel('Mean Error')

        xlim = axes[index, 1].get_xlim()
        ylim = axes[index, 1].get_ylim()
        axes[index, 1].plot(
            xlim, xlim,
            color=[0, 0, 0], linestyle='dotted'
            )
        axes[index, 1].set_xlim(xlim)
        axes[index, 1].set_ylim(ylim)

        xlim = axes[index, 2].get_xlim()
        ylim = axes[index, 2].get_ylim()
        axes[index, 2].plot(
            xlim, xlim,
            color=[0, 0, 0], linestyle='dotted'
            )
        axes[index, 2].set_xlim(xlim)
        axes[index, 2].set_ylim(ylim)
    axes[0, 0].legend(frameon=False)
    axes[n_outputs - 1, 0].set_xlabel('Normalized Residuals')
    axes[n_outputs - 1, 1].set_xlabel('Uncertainty Estimate')
    axes[n_outputs - 1, 2].set_xlabel('Error')
    fig.tight_layout()
    fig.savefig(save_to_name)
    plt.close()

def evaluate_tetragonal_large_errors(data):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_volume_scale = 1000

    data = data[~data['augmented']]
    unit_cell_true = np.stack(data[self.unit_cell_key])[:, self.data_params['y_indices']]
    unit_cell_pred = np.stack(data[f'{self.unit_cell_key}_pred'])
    unit_cell_mse = 1/self.data_params['n_outputs'] * np.linalg.norm(unit_cell_pred - unit_cell_true, axis=1)**2
    large_errors = unit_cell_mse > np.sort(unit_cell_mse)[int(0.75 * data.shape[0])]
    N_small = np.sum(~large_errors)
    N_large = np.sum(large_errors)

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    # volume
    volume = np.array(data[self.volume_key]) / plot_volume_scale
    axes[0, 0].boxplot([volume[~large_errors], volume[large_errors]])
    axes[0, 0].set_title(f'Volume (x{plot_volume_scale})')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xticks([1, 2])
    axes[0, 0].set_xticklabels(['Small err', 'Large err'])

    # minimum unit cell
    uc_min = unit_cell_true.min(axis=1)
    axes[0, 1].boxplot([uc_min[~large_errors], uc_min[large_errors]])
    axes[0, 1].set_title('Minimum unit cell')
    axes[0, 1].set_xticks([1, 2])
    axes[0, 1].set_xticklabels(['Small err', 'Large err'])

    # maximum unit cell
    uc_max = unit_cell_true.max(axis=1)
    axes[0, 2].boxplot([uc_max[~large_errors], uc_max[large_errors]])
    axes[0, 2].set_title('Maximum unit cell')
    axes[0, 2].set_xticks([1, 2])
    axes[0, 2].set_xticklabels(['Small err', 'Large err'])

    # dominant zone ratio
    ratio =  uc_max / uc_min
    axes[0, 3].boxplot([ratio[~large_errors], ratio[large_errors]])
    axes[0, 3].set_title('Maximum / Minimum\nunit cell ratio')
    axes[0, 3].set_xticks([1, 2])
    axes[0, 3].set_xticklabels(['Small err', 'Large err'])

    # variation in unit cell sizes
    #   (a - b)**2 + (a - c)**2 + (b - c)**2
    variation = \
        ((unit_cell_true[:, 0] - unit_cell_true[:, 1]) / (0.5*(unit_cell_true[:, 0] + unit_cell_true[:, 1])))**2
    axes[1, 0].boxplot([variation[~large_errors], variation[large_errors]])
    axes[1, 0].set_title('Variation in unit cell')
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(['Small err', 'Large err'])

    # order of unit cell axis sizes
    order_small = np.argsort(unit_cell_true[~large_errors], axis=1)
    order_large = np.argsort(unit_cell_true[large_errors], axis=1)
    # order: [[shortest index, middle index, longest index], ... ]
    proportions_small = np.zeros((2, 2))
    proportions_large = np.zeros((2, 2))
    for length_index in range(2):
        for uc_index in range(2):
            proportions_small[length_index, uc_index] = np.sum(order_small[:, length_index] == uc_index)
            proportions_large[length_index, uc_index] = np.sum(order_large[:, length_index] == uc_index)
    proportions_small = proportions_small / N_small
    proportions_large = proportions_large / N_large
    axes[1, 1].bar([0, 1], proportions_small[0], color=colors[0], label='Small err')
    axes[1, 1].bar([3, 4], proportions_small[1], color=colors[0])

    axes[1, 1].bar([0, 1], proportions_large[0], color=colors[1], alpha=0.5, label='Large err')
    axes[1, 1].bar([3, 4], proportions_large[1], color=colors[1], alpha=0.5)

    axes[1, 1].set_title('Order of axes lengths')
    axes[1, 1].set_xticks([0.5, 2.5])
    axes[1, 1].set_xticklabels(['Shortest', 'Longest'])
    axes[1, 1].legend(frameon=False)

    # centering
    primitive = np.array(data['bravais_lattice'] == 'tP')
    body_centered = np.array(data['bravais_lattice'] == 'tI')

    # fraction centered
    centered_frac_small = np.sum(~primitive[~large_errors]) / N_small
    centered_frac_large = np.sum(~primitive[large_errors]) / N_large
    axes[1, 2].bar([0, 2], [centered_frac_small, centered_frac_large])
    axes[1, 2].set_xticks([0, 2])
    axes[1, 2].set_xticklabels(['Small error', 'Large error'])
    axes[1, 2].set_title('Fraction centered')

    # fraction Bravais lattice
    tI_frac_small = np.sum(body_centered[~large_errors]) / N_small
    tI_frac_large = np.sum(body_centered[large_errors]) / N_large

    axes[1, 3].bar(0, tI_frac_small, color=colors[0], label='Small err')
    axes[1, 3].bar(0, tI_frac_large, color=colors[1], alpha=0.5, label='Large err')
    axes[1, 3].set_xticks([0])
    axes[1, 3].set_xticklabels(['tI'])
    axes[1, 3].set_title('Bravais Lattice')

    fig.tight_layout()
    fig.savefig(self.save_to['results'] + '_tetragonal_error_eval.png')
    plt.close()

def evaluate_orthorhombic_large_errors(self):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_volume_scale = 1000
    data = self.data[~self.data['augmented']]
    unit_cell_true = np.stack(data[self.unit_cell_key])[:, self.data_params['y_indices']]
    unit_cell_pred = np.stack(data[f'{self.unit_cell_key}_pred'])
    unit_cell_mse = 1/self.data_params['n_outputs'] * np.linalg.norm(unit_cell_pred - unit_cell_true, axis=1)**2
    large_errors = unit_cell_mse > np.sort(unit_cell_mse)[int(0.75 * data.shape[0])]
    N_small = np.sum(~large_errors)
    N_large = np.sum(large_errors)

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    # volume
    volume = np.array(data[self.volume_key]) / plot_volume_scale
    axes[0, 0].boxplot([volume[~large_errors], volume[large_errors]])
    axes[0, 0].set_title(f'Volume (x{plot_volume_scale})')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xticks([1, 2])
    axes[0, 0].set_xticklabels(['Small err', 'Large err'])

    # minimum unit cell
    uc_min = unit_cell_true.min(axis=1)
    axes[0, 1].boxplot([uc_min[~large_errors], uc_min[large_errors]])
    axes[0, 1].set_title('Minimum unit cell')
    axes[0, 1].set_xticks([1, 2])
    axes[0, 1].set_xticklabels(['Small err', 'Large err'])

    # maximum unit cell
    uc_max = unit_cell_true.max(axis=1)
    axes[0, 2].boxplot([uc_max[~large_errors], uc_max[large_errors]])
    axes[0, 2].set_title('Maximum unit cell')
    axes[0, 2].set_xticks([1, 2])
    axes[0, 2].set_xticklabels(['Small err', 'Large err'])

    # dominant zone ratio
    ratio =  uc_max / uc_min
    axes[0, 3].boxplot([ratio[~large_errors], ratio[large_errors]])
    axes[0, 3].set_title('Maximum / Minimum\nunit cell ratio')
    axes[0, 3].set_xticks([1, 2])
    axes[0, 3].set_xticklabels(['Small err', 'Large err'])

    # variation in unit cell sizes
    #   (a - b)**2 + (a - c)**2 + (b - c)**2
    variation = \
        ((unit_cell_true[:, 0] - unit_cell_true[:, 1]) / (0.5*(unit_cell_true[:, 0] + unit_cell_true[:, 1])))**2 \
        + ((unit_cell_true[:, 0] - unit_cell_true[:, 2]) / (0.5*(unit_cell_true[:, 0] + unit_cell_true[:, 2])))**2 \
        + ((unit_cell_true[:, 1] - unit_cell_true[:, 2]) / (0.5*(unit_cell_true[:, 1] + unit_cell_true[:, 2])))**2
    axes[1, 0].boxplot([variation[~large_errors], variation[large_errors]])
    axes[1, 0].set_title('Variation in unit cell')
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(['Small err', 'Large err'])

    # order of unit cell axis sizes
    order_small = np.argsort(unit_cell_true[~large_errors], axis=1)
    order_large = np.argsort(unit_cell_true[large_errors], axis=1)
    # order: [[shortest index, middle index, longest index], ... ]
    proportions_small = np.zeros((3, 3))
    proportions_large = np.zeros((3, 3))
    for length_index in range(3):
        for uc_index in range(3):
            proportions_small[length_index, uc_index] = np.sum(order_small[:, length_index] == uc_index)
            proportions_large[length_index, uc_index] = np.sum(order_large[:, length_index] == uc_index)
    proportions_small = proportions_small / N_small
    proportions_large = proportions_large / N_large
    axes[1, 1].bar([0, 1, 2], proportions_small[0], color=colors[0], label='Small err')
    axes[1, 1].bar([4, 5, 6], proportions_small[1], color=colors[0])
    axes[1, 1].bar([8, 9, 10], proportions_small[2], color=colors[0])

    axes[1, 1].bar([0, 1, 2], proportions_large[0], color=colors[1], alpha=0.5, label='Large err')
    axes[1, 1].bar([4, 5, 6], proportions_large[1], color=colors[1], alpha=0.5)
    axes[1, 1].bar([8, 9, 10], proportions_large[2], color=colors[1], alpha=0.5)

    axes[1, 1].set_title('Order of axes lengths')
    axes[1, 1].set_xticks([1, 5, 9])
    axes[1, 1].set_xticklabels(['Shortest', 'Middle', 'Longest'])
    axes[1, 1].legend(frameon=False)

    # centering
    primitive = np.array(data['bravais_lattice'] == 'oP')
    base_centered = np.array(data['bravais_lattice'] == 'oC')
    body_centered = np.array(data['bravais_lattice'] == 'oI')
    face_centered = np.array(data['bravais_lattice'] == 'oF')

    # fraction centered
    centered_frac_small = np.sum(~primitive[~large_errors]) / N_small
    centered_frac_large = np.sum(~primitive[large_errors]) / N_large
    axes[1, 2].bar([0, 2], [centered_frac_small, centered_frac_large])
    axes[1, 2].set_xticks([0, 2])
    axes[1, 2].set_xticklabels(['Small error', 'Large error'])
    axes[1, 2].set_title('Fraction centered')

    # fraction Bravais lattice
    oC_frac_small = np.sum(base_centered[~large_errors]) / N_small
    oC_frac_large = np.sum(base_centered[large_errors]) / N_large

    oI_frac_small = np.sum(body_centered[~large_errors]) / N_small
    oI_frac_large = np.sum(body_centered[large_errors]) / N_large

    oF_frac_small = np.sum(face_centered[~large_errors]) / N_small
    oF_frac_large = np.sum(face_centered[large_errors]) / N_large

    frac_small = [oC_frac_small, oI_frac_small, oF_frac_small]
    frac_large = [oC_frac_large, oI_frac_large, oF_frac_large]

    axes[1, 3].bar([0, 1, 2], frac_small, color=colors[0], label='Small err')
    axes[1, 3].bar([0, 1, 2], frac_large, color=colors[1], alpha=0.5, label='Large err')
    axes[1, 3].set_xticks([0, 1, 2])
    axes[1, 3].set_xticklabels(['oC', 'oI', 'oF'])
    axes[1, 3].set_title('Bravais Lattice')

    fig.tight_layout()
    fig.savefig(self.save_to['results'] + '_orthorhombic_error_eval.png')
    plt.close()
