import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os


def plot_distance(sorted_results, vlmc, gc_distance_function, metadata, out_dir, add_gc=True, add_sequence_lengths=False):
  fig, ax = plt.subplots(1, sharex='col', figsize=(30, 20), dpi=80)
  ax.set_title(metadata[vlmc.name]['species'], fontsize=30)
  ax.set_xlim(-1, len(sorted_results))
  ax.grid(color='#cccccc', linestyle='--', linewidth=1)
  ax.set_xlabel('Species')
  ax.set_ylabel('Distances')

  species_names = ["{}".format(metadata[v.name]['species']) for _, v in sorted_results]
  plt.xticks(np.arange(len(sorted_results)), rotation=30, fontsize=16)
  ax.set_xticklabels(species_names, ha="right")

  legend_markers = [
      legend_marker('Same order', 'o', '#ff007f'),
      legend_marker('Same family', 'o', '#ff7f00'),
      legend_marker('Same subfamily', 'o', '#7fff00'),
      legend_marker('Same genus', 'o', '#007fff'),
      legend_marker('Same host', '*', '#000000'),
      legend_marker('Distance', '.', '#000000')
  ]

  if add_gc:
    gc_legend_marker = legend_marker('GC difference', '.', '#999999', 'dashed')
    legend_markers.append(gc_legend_marker)

    gc_distances = [(gc_distance_function.distance(vlmc, v), v) for _, v in sorted_results]
    plot_distance_(ax, gc_distances, vlmc, metadata, '#999999', '.', 'dashed')

  if add_sequence_lengths:
    sequence_length_legend_marker = legend_marker('Sequence length', 'x', '#999999', 'dotted')
    legend_markers.append(sequence_length_legend_marker)

    plot_sequence_lengths(ax, sorted_results, vlmc, metadata)

  plot_distance_(ax, sorted_results, vlmc, metadata, '#000000', '.', 'solid')

  # plt.xticks(np.arange(len(sorted_results)), fontsize=20)

  ax.legend(handles=legend_markers, fontsize=24)

  out_file = os.path.join(out_dir, vlmc.name + "-distance.pdf")
  fig.tight_layout(pad=2.2)
  fig.savefig(out_file, dpi='figure', format='pdf')
  plt.close(fig)


def legend_marker(label, marker, color, linestyle=None):
  return Line2D([0], [0], marker=marker, markersize=16, linestyle=linestyle,
                markerfacecolor=color, color=color, label=label)


def plot_sequence_lengths(ax, sorted_results, vlmc, metadata):
  seq_color = '#999999'
  seq_ax = ax.twinx()
  seq_ax.set_ylabel('Sequence lengths', color=seq_color)
  seq_ax.tick_params(axis='y', labelcolor=seq_color)

  lengths = [(i, metadata[v.name]['sequence_length']) for i, (_, v) in enumerate(sorted_results)]

  seq_ax.set_ylim(-20000, max([l for _, l in lengths]) + 20000)

  plot_tuple_list(seq_ax, lengths, seq_color, 'x', 'dotted')


def plot_distance_(ax, sorted_results, vlmc, metadata, line_c, line_marker, linestyle):
  family = [(i, d) for i, (d, v) in enumerate(sorted_results)
            if _equal_and_not_empty(metadata, v, vlmc, 'family') and
            _not_equal_or_empty(metadata, v, vlmc, 'genus')]

  genus = [(i, d) for i, (d, v) in enumerate(sorted_results)
           if _equal_and_not_empty(metadata, v, vlmc, 'genus')]

  baltimore = [(i, d) for i, (d, v) in enumerate(sorted_results)
               if _equal_and_not_empty(metadata, v, vlmc, 'baltimore')]

  every = [(i, d) for i, (d, v) in enumerate(sorted_results)]

  plot_tuple_list(ax, every, line_c, line_marker, linestyle)
  scatter_tuple_list(ax, family, '#ff7f00', 'o')
  scatter_tuple_list(ax, genus, '#007fff', 'o')
  # scatter_tuple_list(ax, baltimore, line_c, '*')
  # scatter_tuple_list(ax, hosts, line_c, '*')


def host_in(hosts_left, hosts_right):
  single_hosts = hosts_left.split(", ")
  for host in single_hosts:
    if host in hosts_right:
      return True
  return False


def _not_equal_or_empty(metadata, v, vlmc, taxonomy):
  return (metadata[v.name][taxonomy] != metadata[vlmc.name][taxonomy] or
          metadata[v.name][taxonomy] == '')


def _equal_and_not_empty(metadata, v, vlmc, taxonomy):
  return (metadata[v.name][taxonomy] == metadata[vlmc.name][taxonomy] and
          metadata[v.name][taxonomy] != '')


def plot_tuple_list(ax, tuple_list, color, marker, linestyle):
  xs = [x for x, _ in tuple_list]
  ys = [y for _, y in tuple_list]
  ax.plot(xs, ys, markersize=20, color=color, marker=marker, linestyle=linestyle)


def scatter_tuple_list(ax, tuple_list, c, marker):
  xs = [x for x, _ in tuple_list]
  ys = [y for _, y in tuple_list]
  ax.scatter(xs, ys, s=600, c=c, marker=marker)


def update_gc_box_data(vlmc, index, sorted_results, all_gc_differences, gc_distance_function):
  gc_distances = [gc_distance_function.distance(vlmc, v) for _, v in sorted_results]
  for i, gc_distance in enumerate(gc_distances):
    all_gc_differences[index, i] = gc_distance


def update_metadata_box(vlmc, index, sorted_results, all_meta_orders, metadata, meta_key):
  for i, (_, other) in enumerate(sorted_results):
    if metadata[other.name][meta_key] == metadata[vlmc.name][meta_key]:
      all_meta_orders[index, i] = 1
    else:
      all_meta_orders[index, i] = 0


def plot_gc_box(all_gc_differences, number_of_bins, out_dir):
  gc_binned = all_gc_differences.T.reshape(-1)
  gc_binned = np.array_split(gc_binned, number_of_bins, axis=0)

  fig, ax = plt.subplots(1, figsize=(80, 20), dpi=80)
  boxplot(ax, gc_binned, "GC-difference")

  out_file = os.path.join(out_dir, "box-gc.pdf")
  fig.savefig(out_file, dpi='figure', format='pdf')


def plot_cummlative_box(all_meta_orders, number_of_bins, meta_name, out_dir):
  meta_binned = bin_cummulative(all_meta_orders, number_of_bins)

  fig, ax = plt.subplots(1, figsize=(80, 20), dpi=80)
  boxplot(ax, meta_binned, "Cummulative percent of {} captured".format(meta_name), (0, 1.1))

  out_file = os.path.join(out_dir, "box-{}.pdf".format(meta_name))
  fig.savefig(out_file, dpi='figure', format='pdf')


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
