import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os


def draw_gc_plot(sorted_results, vlmc, gc_distance_function, distance_ax, gc_ax):
  gc_distances = [(gc_distance_function.distance(vlmc, v), v) for _, v in sorted_results]

  draw_graph(sorted_results, distance_ax, "Distance")
  draw_graph(gc_distances, gc_ax, "GC-difference")


def draw_graph(distances, ax, title):
  ax.set_title(title, fontsize=24)
  distances = [d for d, _ in distances]
  ax.plot(distances)


def plot_distance(sorted_results, vlmc, gc_distance_function, metadata, out_dir):
  fig, ax = plt.subplots(1, sharex='col', figsize=(30, 20), dpi=80)
  ax.set_title(metadata[vlmc.name]['species'], fontsize=30)
  ax.set_xlim(-1, len(sorted_results))
  ax.grid(color='#cccccc', linestyle='--', linewidth=1)

  species_names = ["{}".format(metadata[v.name]['species']) for _, v in sorted_results]

  legend_markers = [Line2D([0], [0], marker='o', markersize=16, markeredgecolor='#ff7f00',
                           markerfacecolor='#ff7f00', label='Same family'),
                    Line2D([0], [0], marker='o', markersize=16, markeredgecolor='#007fff',
                           markerfacecolor='#007fff', label='Same genus'),
                    Line2D([0], [0], marker='.', markersize=16, linestyle='dashed',
                           markerfacecolor='#999999', color='#999999', label='GC-difference'),
                    Line2D([0], [0], marker='.', markersize=16,
                           markerfacecolor='#000000', color='#000000', label='Distance')
                    ]
  ax.legend(handles=legend_markers, fontsize=24)

  plot_distance_(ax, sorted_results, vlmc, metadata, '#000000', 'solid')

  gc_distances = [(gc_distance_function.distance(vlmc, v), v) for _, v in sorted_results]
  plot_distance_(ax, gc_distances, vlmc, metadata, '#999999', 'dashed')

  ax.set_xticklabels(species_names, ha="right")
  plt.xticks(np.arange(len(sorted_results)), rotation=30, fontsize=16)

  # plt.xticks(np.arange(len(sorted_results)), fontsize=20)

  out_file = os.path.join(out_dir, vlmc.name + "-distance.pdf")
  fig.savefig(out_file, dpi='figure', format='pdf')
  plt.close(fig)


def plot_distance_(ax, sorted_results, vlmc, metadata, line_c, linestyle):
  family = [(i, d) for i, (d, v) in enumerate(sorted_results)
            if metadata[v.name]['family'] == metadata[vlmc.name]['family'] and
            metadata[v.name]['genus'] != metadata[vlmc.name]['genus']]

  genus = [(i, d) for i, (d, v) in enumerate(sorted_results)
           if metadata[v.name]['genus'] == metadata[vlmc.name]['genus']]

  other = [(i, d) for i, (d, v) in enumerate(sorted_results)
           if metadata[v.name]['genus'] != metadata[vlmc.name]['genus'] and
           metadata[v.name]['family'] != metadata[vlmc.name]['family']]

  every = [(i, d) for i, (d, v) in enumerate(sorted_results)]

  plot_tuple_list(ax, every, line_c, '.', linestyle)
  scatter_tuple_list(ax, family, '#ff7f00', 'o')
  scatter_tuple_list(ax, genus, '#007fff', 'o')
  # scatter_tuple_list(ax, other, 'k', '.')


def plot_tuple_list(ax, tuple_list, color, marker, linestyle):
  xs = [x for x, _ in tuple_list]
  ys = [y for _, y in tuple_list]
  ax.plot(xs, ys, markersize=20, color=color, marker=marker, linestyle=linestyle)


def scatter_tuple_list(ax, tuple_list, c, marker):
  xs = [x for x, _ in tuple_list]
  ys = [y for _, y in tuple_list]
  ax.scatter(xs, ys, s=600, c=c, marker=marker)


def update_box_plot_data(vlmc, index, sorted_results, all_gc_differences,
                         all_family_orders, all_genus_orders, gc_distance_function, metadata):
  gc_distances = [gc_distance_function.distance(vlmc, v) for _, v in sorted_results]
  for i, gc_distance in enumerate(gc_distances):
    all_gc_differences[index, i] = gc_distance

  for i, (_, other) in enumerate(sorted_results):
    if metadata[other.name]['family'] == metadata[vlmc.name]['family']:
      all_family_orders[index, i] = 1
    else:
      all_family_orders[index, i] = 0

  for i, (_, other) in enumerate(sorted_results):
    if metadata[other.name]['genus'] == metadata[vlmc.name]['genus']:
      all_genus_orders[index, i] = 1
    else:
      all_genus_orders[index, i] = 0


def draw_box_plot(all_gc_differences, all_family_orders, all_genus_orders, number_of_bins, out_dir):
  gc_binned = all_gc_differences.T.reshape(-1)
  gc_binned = np.array_split(gc_binned, number_of_bins, axis=0)

  family_binned = bin_cummulative(all_family_orders, number_of_bins)
  genus_binned = bin_cummulative(all_genus_orders, number_of_bins)

  fig, [gc_ax, family_ax, genus_ax] = plt.subplots(3, figsize=(30, 20), dpi=80)

  boxplot(gc_ax, gc_binned, "GC-difference")
  boxplot(family_ax, family_binned, "Cummulative percent of family captured", (0, 1.1))
  boxplot(genus_ax, genus_binned, "Cummulative percent of genus captured", (0, 1.1))

  out_file = os.path.join(out_dir, "box.pdf")
  fig.savefig(out_file, dpi='figure', format='pdf')
  plt.show()


def boxplot(ax, data, title, ylim=None):
  ax.set_title(title)
  ax.boxplot(data)
  ax.grid(color='#cccccc', linestyle='--', linewidth=1)
  if not ylim is None:
    ax.set_ylim(ylim)


def bin_cummulative(input, number_of_bins):
  cumulative = np.cumsum(input, axis=1)
  last_column = cumulative[:, -1].reshape(cumulative.shape[0], 1)
  cumulative_distribution = cumulative / last_column

  cumlative_transpose = cumulative_distribution.T
  binned = np.array_split(cumlative_transpose, number_of_bins, axis=0)
  max_of_each_vlmc_binned = [np.max(bin_, axis=0) for bin_ in binned]
  return max_of_each_vlmc_binned
