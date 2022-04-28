# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:56:19 2020

This python file contains functions to plot the figures displayed in
"Multiscale communication in cortico-cortical networks".

@author: Vincent Bazinet
"""

import os
import bct
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, rankdata, pearsonr, spearmanr
from netneurotools.plotting import plot_fsaverage
from netneurotools.metrics import communicability_wei

from palettable.colorbrewer.sequential import YlGn_9, YlGnBu_9
from palettable.colorbrewer.sequential import YlOrRd_9, YlOrBr_9
from palettable.colorbrewer.diverging import Spectral_11, Spectral_4
from palettable.colorbrewer.diverging import Spectral_7, RdBu_11_r
from palettable.colorbrewer.diverging import Spectral_4_r

'''
MAIN FIGURE FUNCTIONS
'''


def figure_1(data, specify, t_ids_a, t_ids_c, method='laplacian',
             panels='all', show=True, save=False, save_path=None):
    '''
    Function to create Figure 1

    Parameters
    ----------
    data : dict
        Dictionary of the data to be used to generate the figures. If the
        required data is not found in this dictionary, the item of figure
        1 that requires this data will not be created and a message
        will be printed.
    specify : dict
        Dictionary containing information about the nodes that will be
        shown in panel a (see main_script.py for an example).
    t_ids_a: List
        List of time points indices indicating the time points at which
        the transition probabilities will be shown for our nodes of interest
        (panel a).
    t_ids_c: List
        List of time points indices indicating the time points at which
        the multiscale centrality distribution will be plotted on the surface
        of the brain (panel c).
    method : str
        Method used to compute the transition probabilities. Here, we specify
        either laplacian or pagerank. If transition probabilities were
        generated using a laplacian matrix, the contrained walks in panel a
        will be generated as normal random walks regardless of the type of
        Laplacian matrix that was actually used.
    panels : str or list
        List of the panels of Figure 1 that we want to create. If we want
        to create all of them, use 'all'. Otherwise, individual panels can
        be specified. For example, we could have panels=['a'] or
        panels=['a', 'b'].
    show : Boolean
        If True, the figures will be displayed. If not, the figures will
        bot be displayed.
    save : Boolean
        If True, the figures will be saved in the folder specified by the
        save_path parameter.
    save_path : str
        Path of the folder in which the figures will be saved.
    '''

    if show is False:
        plt.ioff()

    if panels == 'all':
        panels = ['a', 'b', 'c']

    if 'a' in panels:

        required_entries = ['sc', 'coords', 't_points']
        requirements = check_requirements(data, required_entries)

        if requirements is True:

            walk_type = method
            if method == 'laplacian':
                walk_type = 'normal'

            for i, seed_node in enumerate(specify['ID']):
                for t_id_a in t_ids_a:

                    walk_type = method
                    if method == 'laplacian':
                        walk_type = 'normal'
                    fig = plot_constrained_walk(data,
                                                seed_node,
                                                data['t_points'][t_id_a],
                                                walk_type=walk_type,
                                                color=specify['colors'][i])
                    plt.title(specify['labels'][i])
                    if save:
                        fig_name = ("constrained_walk_" +
                                    str(seed_node) + "_" +
                                    str(round(data['t_points'][t_id_a], 0)) +
                                    ".png")
                        fig.savefig(os.path.join(save_path, fig_name))

    if 'b' in panels:

        required_entries = ['sc', 't_points', 'cmulti']
        requirements = check_requirements(data, required_entries)

        if requirements is True:

            n = len(data['sc'])
            fig = plt.figure(figsize=(9, 3))
            ax = fig.add_subplot(111)
            for i in range(n):
                ax.plot(data['t_points'], data['cmulti'][:, i], c='lightgray')
            for i, ID in enumerate(specify["ID"]):
                ax.plot(data['t_points'],
                        data['cmulti'][:, ID],
                        c=specify["colors"][i],
                        label=specify["labels"][i])
            ax.legend()
            ax.set_xlabel('t')
            ax.set_ylabel('c_multi')
            if method == 'laplacian':
                ax.set_xscale('log')

            if save:
                figure_name = 'c_multi_trajectory.png'
                fig.savefig(os.path.join(save_path, figure_name))

    if 'c' in panels:

        required_entries = ['sc', 't_points', 'cmulti', 'lhannot', 'rhannot',
                            'noplot', 'order']
        requirements = check_requirements(data, required_entries)

        if requirements is True:

            cmaps = [YlOrRd_9, YlOrBr_9, YlGn_9, YlGnBu_9]

            n = len(data['sc'])
            fig = plt.figure(figsize=(9, 3))
            ax = fig.add_subplot(111)
            for i in range(n):
                ax.plot(data['t_points'],
                        data['cmulti'][:, i],
                        c='lightgray',
                        zorder=0)
            for i, t in enumerate(t_ids_c):
                c = rankdata(data['cmulti'][t, :])
                plt.scatter(np.zeros((n))+data['t_points'][t],
                            data['cmulti'][t, :],
                            marker='s', c=c,
                            cmap=cmaps[i].mpl_colormap,
                            rasterized=True,
                            zorder=1)
            if method == 'laplacian':
                ax.set_xscale('log')
            ax.set_ylabel('c_multi')
            ax.set_xlabel('t')

            if save:
                figure_name = 'c_multi_trajectory_2.png'
                fig.savefig(os.path.join(save_path, figure_name))

            for i, t in enumerate(t_ids_c):

                scores = rankdata(data['cmulti'][t, :])
                im = plot_fsaverage(scores,
                                    lhannot=data['lhannot'],
                                    rhannot=data['rhannot'],
                                    noplot=data['noplot'],
                                    order=data['order'],
                                    views=['lateral', 'm'],
                                    vmin=np.amin(scores),
                                    vmax=np.amax(scores),
                                    colormap=cmaps[i].mpl_colormap)

                if save:
                    figure_name = 'cmulti_surface_'+str(t)+'.png'
                    im.save_image(os.path.join(save_path, figure_name),
                                  mode='rgba')

    if show is False:
        plt.close('all')
        plt.ion()


def figure_2(data, method='laplacian', panels='all', show=True, save=False,
             save_path=None):
    '''
    Function to create Figure 2

    Parameters
    ----------
    data : dict
        Dictionary of the data to be used to generate the figures. If the
        required data is not found in this dictionary, the item of figure
        1 that requires this data will not be created and a message
        will be printed.
    method : str
        Method used to compute the transition probabilities. The purpose of
        this parameter is to choose whether the x_scale of our figures should
        be linear or logarithmic.
    panels : str or list
        List of the panels of Figure 1 that we want to create. If we want
        to create all of them, use 'all'. Otherwise, individual panels can
        be specified. For example, we could have panels=['a'] or
        panels=['a', 'b'].
    show : Boolean
        If True, the figures will be displayed. If not, the figures will
        bot be displayed.
    save : Boolean
        If True, the figures will be saved in the folder specified by the
        save_path parameter.
    save_path : str
        Path of the folder in which the figures will be saved.
    '''

    if show is False:
        plt.ioff()

    if panels == 'all':
        panels = ['a', 'b', 'c']

    n = len(data['sc'])
    k = len(data['t_points'])

    # Compute optimal centrality scale for individual nodes
    opti = np.argmax(data['cmulti'], axis=0)

    if 'a' in panels:

        required_entries = ['t_points', 'cmulti', 'sc', 'lhannot', 'rhannot',
                            'noplot', 'order']
        requirements = check_requirements(data, required_entries)
        if requirements is True:

            norm = plt.Normalize(np.amin(opti), np.amax(opti))
            optimal_colors = Spectral_4.mpl_colormap(norm(opti))

            fig = plt.figure(figsize=(9, 3))
            ax = fig.add_subplot(111)
            for i in range(n):
                ax.plot(data['t_points'],
                        data['cmulti'][:, i],
                        c=optimal_colors[i, :])

            ax.set_xlabel('t')
            ax.set_ylabel('c_multi')
            if method == 'laplacian':
                ax.set_xscale('log')

            if save:
                fig_name = "cmulti_trajectory_colored.png"
                plt.savefig(os.path.join(save_path, fig_name))

            log_topti = np.log10(data['t_points'][opti])
            im = plot_fsaverage(log_topti,
                                lhannot=data['lhannot'],
                                rhannot=data['rhannot'],
                                noplot=data['noplot'],
                                order=data['order'],
                                views=['lateral', 'm'],
                                vmin=np.amin(log_topti),
                                vmax=np.amax(log_topti),
                                colormap=Spectral_4.mpl_colormap)

            if save:
                figure_name = 'optimal_brain_surface.png'
                im.save_image(os.path.join(save_path, figure_name),
                              mode='rgba')

    if 'b' in panels:

        required_entries = ['rsn', 'rsn_names', 'cmulti', 't_points']
        requirements = check_requirements(data, required_entries)
        if requirements:

            opti_rsn = []
            median_rsn = np.zeros((7))
            for i in range(7):
                rsn_ids = np.where(data['rsn'] == i+1)[0]
                opti_rsn.append(data['t_points'][opti[rsn_ids]])
                median_rsn[i] = np.median(opti[rsn_ids])

            rsn_average = np.zeros((k, 7))
            for i in range(k):
                for j in range(7):
                    rsn_ids = np.where(data['rsn'] == j+1)[0]
                    rsn_average[i, j] = np.mean(data['cmulti'][i, rsn_ids])

            # Sort resting-state networks according to median optimal scores.
            rsn_order = np.argsort(median_rsn)
            opti_rsn_sorted = np.array(opti_rsn)[rsn_order].tolist()
            rsn_names_sorted = np.array(data['rsn_names'])[rsn_order].tolist()

            colormap = Spectral_7
            c_nb = [0, 1, 2, 3, 4, 5, 6]

            # optimal_rsn_violin
            plt.figure(figsize=(6, 3))
            sns.violinplot(data=opti_rsn_sorted,
                           palette=np.array(colormap.hex_colors)[c_nb],
                           orient='v')
            plt.ylabel("t_opti")
            plt.xticks(np.arange(0, 7), rsn_names_sorted)
            if method == "laplacian":
                y_min = np.amin(data['t_points'])
                y_max = np.amax(data['t_points'])
                plt.yticks([y_min, y_max],
                           [round(y_min, 0), round(y_max, 0)])

            if save:
                figure_name = 'optimal_rsn_violin'
                plt.savefig(os.path.join(save_path, figure_name))

            # optimal_rsn_average
            plt.figure(figsize=(6, 3))
            for j in range(7):
                plot_color = colormap.hex_colors[c_nb[j]]
                idx = np.argsort(median_rsn)[j]
                plt.plot(data['t_points'], rsn_average[:, idx],
                         label=data['rsn_names'][idx], c=plot_color,
                         linewidth=3)
            plt.xlabel("t")
            plt.ylabel('c_multi')
            if method == 'laplacian':
                plt.xscale('log')
            plt.legend()

            if save:
                figure_name = 'optimal_rsn_average'
                plt.savefig(os.path.join(save_path, figure_name))

    if 'c' in panels:

        required_entries = ['ve', 've_names', 't_points', 'cmulti']
        requirements = check_requirements(data, required_entries)
        if requirements:
            opti_ve = []
            median_ve = np.zeros((7))
            for i in range(7):
                ve_ids = np.where(data['ve'] == i+1)[0]
                opti_ve.append(data['t_points'][opti[ve_ids]])
                median_ve[i] = np.median(opti[np.where(data['ve'] == i+1)[0]])

            ve_average = np.zeros((k, 7))
            for i in range(k):
                for j in range(7):
                    ve_ids = np.where(data['ve'] == j+1)[0]
                    ve_average[i, j] = np.mean(data['cmulti'][i, ve_ids])

            # Sort resting-state networks according to median optimal scores.
            ve_order = np.argsort(median_ve)
            opti_ve_sorted = np.array(opti_ve)[ve_order].tolist()
            ve_names_sorted = np.array(data['ve_names'])[ve_order].tolist()

            colormap = Spectral_11
            c_nb = [1, 2, 3, 4, 7, 8, 9]

            # optimal_ve_violin
            plt.figure(figsize=(6, 3))
            sns.violinplot(data=opti_ve_sorted,
                           palette=np.array(colormap.hex_colors)[c_nb],
                           orient='v')
            plt.ylabel("t_opti")
            plt.xticks(np.arange(0, 7), ve_names_sorted)
            if method == "laplacian":
                y_min = np.amin(data['t_points'])
                y_max = np.amax(data['t_points'])
                plt.yticks([y_min, y_max],
                           [round(y_min, 0), round(y_max, 0)])

            if save:
                figure_name = 'optimal_ve_violin'
                plt.savefig(os.path.join(save_path, figure_name))

            # optimal_ve_average
            plt.figure(figsize=(6, 3))
            for j in range(7):
                plot_color = colormap.hex_colors[c_nb[j]]
                idx = np.argsort(median_ve)[j]
                plt.plot(data['t_points'], ve_average[:, idx],
                         label=data['ve_names'][idx], c=plot_color,
                         linewidth=3)
            plt.xlabel("t")
            plt.ylabel('c_multi')
            if method == 'laplacian':
                plt.xscale('log')
            plt.legend()

            if save:
                figure_name = 'optimal_ve_average'
                plt.savefig(os.path.join(save_path, figure_name))

    if show is False:
        plt.close('all')
        plt.ion()


def figure_4(data, specify, t_ids_b, method='laplacian', panels='all',
             show=True, save=False, save_path=None):
    '''
    Function to create Figure 4

    Parameters
    ----------
    data : dict
        Dictionary of the data to be used to generate the figures. If the
        required data is not found in this dictionary, the item of figure
        1 that requires this data will not be created and a message
        will be printed.
    specify : dict
        Dictionary containing information about the trajectories that will be
        shown in panel a (see main_script.py for an example).
    t_ids_b: List
        List of time points indices indicating the time points at which
        the centrality slope distributions will be plotted on the surface
        of the brain (panel b).
    method : str
        Method used to compute the transition probabilities. The purpose of
        this parameter is to choose whether the x_scale of our figures should
        be linear or logarithmic.
    panels : str or list
        List of the panels of Figure 1 that we want to create. If we want
        to create all of them, use 'all'. Otherwise, individual panels can
        be specified. For example, we could have panels=['a'] or
        panels=['a', 'b'].
    show : Boolean
        If True, the figures will be displayed. If not, the figures will
        bot be displayed.
    save : Boolean
        If True, the figures will be saved in the folder specified by the
        save_path parameter.
    save_path : str
        Path of the folder in which the figures will be saved.
    '''

    if show is False:
        plt.ioff()

    if panels == 'all':
        panels = ['a', 'b', 'c']

    n = len(data['sc'])
    k = len(data['t_points'])

    # Slopes of the closeness centrality trajectories
    slopes = np.gradient(data['cmulti'], axis=0)

    if 'a' in panels:

        required_entries = ['t_points', 'cmulti']
        requirements = check_requirements(data, required_entries)
        if requirements:

            node_ids = specify['ID']

            fig = plt.figure(figsize=(9, 3))
            ax = fig.add_subplot(111)
            for i in range(n):
                ax.plot(data['t_points'],
                        data['cmulti'][:, i],
                        c='lightgray')

            abs_max_color = max(-1 * np.amin(slopes), np.amax(slopes))

            for i, id in enumerate(node_ids):
                norm = plt.Normalize(-abs_max_color, abs_max_color)
                slope_colors = RdBu_11_r.mpl_colormap(norm(slopes[:, id]))
                for ii in range(k-1):
                    plt.plot([data['t_points'][ii],
                              data['t_points'][ii+1]],
                             [data['cmulti'][ii, id],
                              data['cmulti'][ii+1, id]],
                             c=slope_colors[ii, :],
                             linewidth=3)

            if method == 'laplacian':
                plt.xscale('log')

            plt.xlabel('t')
            plt.ylabel("c_multi")

            if save:
                figure_name = 'cmulti_with_gradient.png'
                fig.savefig(os.path.join(save_path, figure_name))

    if 'b' in panels:

        required_entries = ['t_points', 'lhannot', 'rhannot', 'noplot',
                            'order']
        requirements = check_requirements(data, required_entries)
        if requirements:

            fig = plt.figure(figsize=(9, 3))
            ax = fig.add_subplot(111)
            for i in range(n):
                ax.plot(data['t_points'],
                        slopes[:, i],
                        c='lightgray',
                        zorder=0)
            for t_id in t_ids_b:
                plt.scatter(np.zeros((n))+data['t_points'][t_id],
                            slopes[t_id, :],
                            marker='s', c=slopes[t_id, :],
                            cmap=RdBu_11_r.mpl_colormap, rasterized=True,
                            zorder=1)

            if method == 'laplacian':
                plt.xscale('log')

            plt.xlabel('t')
            plt.ylabel("slope")

            if save:
                figure_name = 'slopes.png'
                fig.savefig(os.path.join(save_path, figure_name))

            for t_id in t_ids_b:
                im = plot_fsaverage(slopes[t_id, :],
                                    lhannot=data['lhannot'],
                                    rhannot=data['rhannot'],
                                    noplot=data['noplot'],
                                    order=data['order'],
                                    views=['lateral', 'm'],
                                    vmin=np.amin(slopes[t_id, :]),
                                    vmax=np.amax(slopes[t_id, :]),
                                    colormap=RdBu_11_r.mpl_colormap)

                if save:
                    figure_name = ('slopes_brain_surface_' +
                                   str(int(round(data['t_points'][t_id]))) +
                                   '.png')
                    im.save_image(os.path.join(save_path, figure_name),
                                  mode='rgba')

    if 'c' in panels:

        required_entries = ['sc', 't_points', 'ci']
        requirements = check_requirements(data, required_entries)
        if requirements:

            measures = []
            labels = []

            measures.append(np.sum(data['sc'], axis=0))
            labels.append("strength")

            measures.append(-bct.clustering_coef_wu(data['sc']))
            labels.append("clustering(-)")

            for ci in data['ci']:
                measures.append(bct.participation_coef(data['sc'], ci))
                labels.append(("participation (" +
                               str(int(round(ci.max()))) +
                               ")"))

            k = len(data['t_points'])
            m = len(measures)

            corrs = np.zeros((m, k))
            for i in range(m):
                for j in range(k):
                    corrs[i, j] = pearsonr(slopes[j, :], measures[i])[0]

            corr_min = np.amin(corrs)
            corr_max = np.amax(corrs)

            for i in range(m):
                plt.figure()
                plt.imshow(corrs[i, :][np.newaxis, :],
                           cmap=Spectral_4_r.mpl_colormap,
                           vmin=corr_min,
                           vmax=corr_max,
                           aspect=0.1 * k)
                plt.axis('off')
                plt.title(labels[i])

                if save is True:
                    figure_name = "correlations_" + labels[i] + ".png"
                    plt.savefig(os.path.join(save_path, figure_name))

    if show is False:
        plt.close('all')
        plt.ion()


def figure_5(data, method='laplacian', panels='all',
             show=True, save=False, save_path=None):
    '''
    Function to create Figure 5

    Parameters
    ----------
    data : dict
        Dictionary of the data to be used to generate the figures. If the
        required data is not found in this dictionary, the item of figure
        1 that requires this data will not be created and a message
        will be printed.
    method : str
        Method used to compute the transition probabilities. The purpose of
        this parameter is to choose whether the x_scale of our figures should
        be linear or logarithmic.
    panels : str or list
        List of the panels of Figure 1 that we want to create. If we want
        to create all of them, use 'all'. Otherwise, individual panels can
        be specified. For example, we could have panels=['a'] or
        panels=['a', 'b'].
    show : Boolean
        If True, the figures will be displayed. If not, the figures will
        bot be displayed.
    save : Boolean
        If True, the figures will be saved in the folder specified by the
        save_path parameter.
    save_path : str
        Path of the folder in which the figures will be saved.
    '''

    if show is False:
        plt.ioff()

    if panels == 'all':
        panels = ['a', 'b', 'c', 'd', 'e', 'f']

    n = len(data['sc'])
    k = len(data['t_points'])

    # Colors for measures in panels (a) and (b)
    c = Spectral_4.hex_colors

    # Create a mask that ignores nodes on the diagonal and nodes with fc
    # fc score bootstrapped to 0.
    mask = np.zeros((n, n), dtype="bool")
    mask[:] = True
    mask[np.diag_indices(n)] = False
    mask[data['fc'] <= 0] = False

    # Identify single-scale measure to be used
    communicability = communicability_wei(data["sc"])
    single_scale_measures = [data['sc'], -data['sp'], communicability]
    labels = ['SC weights', "(-)shortest path", "communicability"]

    # Compute correlations between FC and single-scale measures
    m = len(single_scale_measures)
    single_scale_corrs = np.zeros((m))
    for i, measure in enumerate(single_scale_measures):
        single_scale_corrs[i] = pearsonr(measure[mask], data['fc'][mask])[0]

    # Compute correlations between FC and neighborhood similarity
    nei_sim_corrs = np.zeros((k))
    for i in range(k):
        nei_sim_corrs[i] = pearsonr(data['nei_sim'][i, :, :][mask],
                                    data['fc'][mask])[0]
    best_n_sim_t = np.argmax(nei_sim_corrs)
    best_n_sim = data['nei_sim'][best_n_sim_t, :, :]

    # Local correlations between neighborhood similarity and fc
    # Compute only if not already computed
    if 'local_fc_rhos' not in data:
        rhos = np.zeros((k, n))
        for i in tqdm.trange(k, desc='local fc-similarity rho\'s'):
            for j in range(n):
                rhos[i, j] = spearmanr(np.delete(data['nei_sim'][i, j, :], j),
                                       np.delete(data["fc"][j, :], j))[0]
        data['local_fc_rhos'] = rhos.copy()
    else:
        rhos = data['local_fc_rhos']

    best_t_id = np.argmax(rhos, axis=0)
    rhos_sorted = rhos[:, np.argsort(best_t_id)]
    rhos_sorted_z = zscore(rhos_sorted, axis=0)

    if 'a' in panels:

        required_entries = ['t_points']
        requirements = check_requirements(data, required_entries)
        if requirements:

            fig = plt.figure(figsize=(4, 4))
            for i in range(m):
                plt.plot([data['t_points'][0], data['t_points'][-1]],
                         [single_scale_corrs[i], single_scale_corrs[i]],
                         c=c[i],
                         label=labels[i])
            plt.plot(data['t_points'],
                     nei_sim_corrs,
                     c=c[i+1],
                     linestyle="dashed",
                     label="neighborhood similarity")

            if method == 'laplacian':
                plt.xscale('log')

            plt.xlabel("t")
            plt.ylabel("Pearson's r")
            plt.legend()

            if save:
                figure_name = 'global_fc_correlations.png'
                fig.savefig(os.path.join(save_path, figure_name))

    if 'b' in panels:

        required_entries = ['fc']
        requirements = check_requirements(data, required_entries)
        if requirements:

            for i in range(m):
                plt.figure(figsize=(2, 2))
                plt.scatter(single_scale_measures[i][mask],
                            data["fc"][mask],
                            c=c[i],
                            s=3,
                            alpha=0.1,
                            rasterized=True)
                p = np.polyfit(single_scale_measures[i][mask],
                               data["fc"][mask],
                               1)
                x_fit = np.array([np.amin(single_scale_measures[i][mask]),
                                  np.amax(single_scale_measures[i][mask])])
                y_fit = x_fit * p[0] + p[1]
                plt.plot(x_fit, y_fit, c="black", linestyle="dashed")
                plt.title("r: "+str(round(single_scale_corrs[i], 4)))

                if save:
                    figure_name = "scatter_fc_" + labels[i] + ".png"
                    plt.savefig(os.path.join(save_path, figure_name))

            plt.figure(figsize=(2, 2))
            plt.scatter(best_n_sim[mask],
                        data["fc"][mask], c=c[i+1],
                        alpha=0.1,
                        s=3, rasterized=True)
            p = np.polyfit(best_n_sim[mask],
                           data["fc"][mask], 1)
            x_fit = np.array([np.amin(best_n_sim[mask]),
                              np.amax(best_n_sim[mask])])
            y_fit = x_fit * p[0] + p[1]
            plt.plot(x_fit, y_fit, c="black", linestyle="dashed")
            plt.title("r: "+str(round(np.amax(nei_sim_corrs), 4)))

            if save:
                figure_name = "scatter_fc_nei_similarity.png"
                plt.savefig(os.path.join(save_path, figure_name))

    if 'c' in panels:

        required_entries = ['sc', 'sp']
        requirements = check_requirements(data, required_entries)
        if requirements:

            hm_labels = ["S", "C", "A", r'$\Phi$']
            masked_measures = [best_n_sim[mask],
                               communicability[mask],
                               data['sc'][mask],
                               -data['sp'][mask]]

            correlations = np.corrcoef(np.array(masked_measures))

            fig, ax = plt.subplots()
            im, cbar = heatmap(correlations, hm_labels, hm_labels, ax=ax,
                               cmap=Spectral_4_r.mpl_colormap,
                               cbarlabel="Pearson's r", vmin=0, vmax=1,
                               aspect="equal", grid_width=3)
            annotate_heatmap(im,
                             valfmt="{x:.2f}",
                             textcolors=["black", "white"])

            if save:
                figure_name = "heatmap_fc_correlations.png"
                fig.savefig(os.path.join(save_path, figure_name))

    if 'd' in panels:

        # Heatmap of best Spearman's correlation across alpha values
        plt.figure()
        m = max(abs(np.percentile(rhos_sorted_z, 2.5)),
                np.percentile(rhos_sorted_z, 97.5)
                )
        plt.imshow(rhos_sorted_z.T,
                   aspect='auto',
                   cmap=RdBu_11_r.mpl_colormap,
                   vmin=-m,
                   vmax=m)
        cbar = plt.colorbar()
        cbar.set_label("spearman's rho (row-standardized)")

        if save is True:
            figure_name = 'heatmap_local_fc_correlations.png'
            plt.savefig(os.path.join(save_path, figure_name))

    if 'e' in panels:

        required_entries = ['t_points', 'lhannot', 'rhannot', 'noplot',
                            'order']
        requirements = check_requirements(data, required_entries)
        if requirements:

            log_best_t = np.log10(data['t_points'][best_t_id])
            im = plot_fsaverage(log_best_t,
                                lhannot=data['lhannot'],
                                rhannot=data['rhannot'],
                                noplot=data['noplot'],
                                order=data['order'],
                                views=['lateral', 'm'],
                                vmin=np.amin(log_best_t),
                                vmax=np.amax(log_best_t),
                                colormap=Spectral_4_r.mpl_colormap)

            if save:
                figure_name = "log_best_t_brain_surface.png"
                im.save_image(os.path.join(save_path, figure_name),
                              mode='rgba')

    if show is False:
        plt.close('all')
        plt.ion()


'''
ADDITIONAL FUNCTIONS
'''


def plot_constrained_walk(data, seed, t, walk_type='normal', color="blue",
                          length=5000, max_size=1000, min_size=0,
                          return_visits=False):
    '''
    Function to display a constrained walk (with discrete transitions)
    on a network.

    Parameters
    ----------
    data : dict
        Dictionary of the data to be used to generate the figures. If the
        required data is not found in this dictionary, the item of figure
        1 that requires this data will not be created and a message
        will be printed.
    seed : int
        Integer ID of the seed from which the walk starts.
    walk_type : str
        Type of the walking process. Either "pagerank", where t
        represents a probability of restart, or "normal", where t
        represents the number of steps taken by the walker.
    t : int
        length (or damping factor if 'pagerank') of the walking process.
    color: color
        Color of the nodes. Must be compatible with matplotlib colors.
    length : int
        Number of iterations of the random walk process.
    max_size : int
        Maximum size of the nodes. The size of a node is an indicator
        of the number of times a random walker travel through the node.
    min_size : int
        Minimum size of the nodes. The size of a node is an indicator of the
        number of times a random walker travel through the node.

    Returns
    -------
    fig: Figure
        Figure instance (matplotlib.pyplot.Figure) associated with the figure.
    visits : (n,) ndarray
        ndarray with the number of time each node has been visited by the
        random walkers.
    '''

    degree = np.sum(data['sc'], axis=0)
    T = (data['sc']/degree[:, np.newaxis]).T
    n = len(T)

    # Compute the transition probabilities of the seed node, with the
    # probability of restart if the walk_type is pagerank.
    T_PR = T[:, seed].copy()
    if walk_type == "pagerank":
        T_PR = t * T_PR
        T_PR[seed] += (1-t)

    # Create a list to store the random paths that will be generated.
    random_paths = []
    random_paths.append([])
    it = 0
    walk_nb = 0
    walk_length = 0
    while it < length:

        # Initialize a new random path starting from the seed
        random_paths[walk_nb].append(seed)

        # Randomly pick a neighbor, based on transition probabilities
        k_new = np.random.choice(np.arange(0, n), p=T_PR)

        if walk_type == "pagerank":

            # Update transition probabilities for this new node
            T_PR = T[:, k_new].copy()
            T_PR = t * T_PR
            T_PR[seed] += (1-t)

            # While the new node is not the seed
            while(k_new != seed):

                # If new node is not the seed, then add this node to the path
                random_paths[walk_nb].append(k_new)
                it += 1

                # Pick a new node, then update probabilities
                k_new = np.random.choice(np.arange(0, n), p=T_PR)
                T_PR = T[:, k_new].copy()
                T_PR = t * T_PR
                T_PR[seed] += (1-t)

            # Once walk has ended, reinstatiate new walk
            random_paths.append([])
            walk_nb += 1
            walk_length = 0

        elif walk_type == "normal":

            # Update transition probabilities for this new node
            T_PR = T[:, k_new].copy()

            # While the walk hasn't reached the specified length...
            while(walk_length < t):

                # Add new node to the path
                random_paths[walk_nb].append(k_new)
                walk_length += 1
                it += 1

                # Pick a new node, then update probabilities
                k_new = np.random.choice(np.arange(0, n), p=T_PR)
                T_PR = T[:, k_new].copy()

        # Once walk has ended, reinstatiate new walk, starting at the seed
        T_PR = T[:, seed].copy()
        if walk_type == "pagerank":
            T_PR = t * T_PR
            T_PR[seed] += (1-t)

        random_paths.append([])
        walk_nb += 1
        walk_length = 0

    # Count the number of visits to each nodes
    visits = np.zeros((n))
    for i in range(len(random_paths)):
        for j in range(len(random_paths[i])):
            visits[random_paths[i][j]] += 1

    # Set alpha of nodes to 1 for all nodes
    a = 1
    # Set size of nodes between min_size and max_size, based on nb of visits
    s = scale(visits, min_size, max_size)

    # Set color of nodes based on if visited or not by random walker
    if isinstance(color, str):
        c = np.zeros((n), dtype="<U10")
        c[:] = "lightgray"
    elif isinstance(color, tuple):
        c = np.zeros((n), len(color))

    if isinstance(color, (str, tuple)):
        c[visits > 0] = color
    elif isinstance(color, (np.ndarray, list)):
        c[visits > 0] = color[np.where(visits > 0)[0]
                              ]
    c[seed] = 'black'

    # Plot the constrained walks on the network
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.scatter(data['coords'][:, 0], data['coords'][:, 1], s=s,
               c=c, zorder=2, alpha=a)

    # plot the walks
    for path in random_paths:
        if len(path) > 1:
            ax.plot(data['coords'][path, 0],
                    data['coords'][path, 1],
                    c="black", linestyle="dashed",
                    linewidth=1, alpha=0.05, zorder=1)

    ax.set_xlim(np.amin(data['coords']) - np.std(data['coords']),
                np.amax(data['coords']) + np.std(data['coords']))
    ax.set_ylim(np.amin(data['coords']) - np.std(data['coords']),
                np.amax(data['coords']) + np.std(data['coords']))

    # set aspect to equal
    ax.set_aspect('equal')

    if return_visits:
        return fig, visits
    else:
        return fig


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", grid_width=3, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Credit : https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html # noqa

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=grid_width)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Credit : https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html # noqa

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def check_requirements(data, required_entries):
    requirements = True
    for entry in required_entries:
        if entry not in data:
            requirements = False

    if requirements is False:
        print(('panel not created. The data dictionary is missing'
               'required entries. The required entries are: ' +
               ', '.join(required_entries)))

    return requirements

def scale(values, amin, amax):
    std = (values - values.min()) / (values.max() - values.min())
    scaled = std * (amax - amin) + amin
    return scaled
