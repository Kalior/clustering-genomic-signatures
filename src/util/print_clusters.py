import networkx as nx


def print_connected_components(G, d, distance_mean, metadata):
  connected_component_metrics = [component_metrics(
      connected, metadata, d) for connected in nx.connected_components(G)]

  output = ["cluster {}:\n".format(i) + component_string(connected, metadata, connected_component_metrics[i])
            for i, connected in enumerate(nx.connected_components(G))]

  print('\n\n'.join(output))

  filtered_metrics = [metrics for metrics, connected in zip(
      connected_component_metrics, nx.connected_components(G)) if len(connected) > 1]

  average_of_same_genus = sum(
      [metrics[0] for metrics in filtered_metrics]) / len(filtered_metrics)
  average_of_same_family = sum(
      [metrics[1] for metrics in filtered_metrics]) / len(filtered_metrics)
  total_average_distance = sum(
      [metrics[2] for metrics in filtered_metrics]) / len(filtered_metrics) / distance_mean

  print("Average percent of same genus in clusters: {:5.5f}\t"
        "Average percent of same family in clusters: {:5.5f}\t"
        "Average distance in clusters: {:5.5f}\t".format(
            average_of_same_genus, average_of_same_family, total_average_distance))

  sorted_sizes = sorted([len(connected) for connected in nx.connected_components(G)])
  print("Cluster sizes " + " ".join([str(i) for i in sorted_sizes]))


def component_metrics(connected, metadata, d):
  percent_of_same_genus = sum(
      [number_in_taxonomy(vlmc, connected, metadata, 'genus') for vlmc in connected]
  ) / (len(connected) * len(connected))

  percent_of_same_family = sum(
      [number_in_taxonomy(vlmc, connected, metadata, 'family') for vlmc in connected]
  ) / (len(connected) * len(connected))

  connected_distances = [d.distance(v1, v2) for v1 in connected for v2 in connected]
  average_distance = sum(connected_distances) / len(connected_distances)

  return percent_of_same_genus, percent_of_same_family, average_distance


def component_string(connected, metadata, metrics):
  output = [output_line(metadata, vlmc) for vlmc in connected]

  metric_string = ("\nPercent of same genus: {:5.5f} \t"
                   "Percent of same family: {:5.5f} \t"
                   "Average distance: {:5.5f}\n".format(
                       metrics[0], metrics[1], metrics[2]))

  return '\n'.join(output) + metric_string


def output_line(metadata, vlmc):
  return "{:>55}  {:20} {:20}".format(
      metadata[vlmc.name]['species'],
      metadata[vlmc.name]['genus'],
      metadata[vlmc.name]['family'])


def number_in_taxonomy(vlmc, vlmcs, metadata, taxonomy):
  number_of_same_taxonomy = len([other for other in vlmcs
                                 if metadata[other.name][taxonomy] == metadata[vlmc.name][taxonomy]])
  return number_of_same_taxonomy
