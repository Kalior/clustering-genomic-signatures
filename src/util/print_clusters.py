import networkx as nx


def print_connected_components(clustering_metrics):
  G = clustering_metrics.G
  connected_component_metrics = [component_metrics(
      connected, clustering_metrics) for connected in nx.connected_components(G)]

  metadata = clustering_metrics.metadata
  output = ["cluster {}:\n".format(i) + component_string(connected, metadata, connected_component_metrics[i])
            for i, connected in enumerate(nx.connected_components(G))]

  print('\n\n'.join(output))

  distance_mean = clustering_metrics.distance_mean
  average_of_same_genus = clustering_metrics.average_percent_same_taxonomy('genus')
  average_of_same_family = clustering_metrics.average_percent_same_taxonomy('family')

  print("Average percent of same genus in clusters: {:5.5f}\t"
        "Average percent of same family in clusters: {:5.5f}\n".format(
            average_of_same_genus, average_of_same_family))

  sorted_sizes = sorted([len(connected) for connected in nx.connected_components(G)])
  print("Cluster sizes " + " ".join([str(i) for i in sorted_sizes]))


def component_metrics(connected_component, clustering_metrics):
  percent_of_same_genus = clustering_metrics.percent_same_taxonomy(connected_component, 'genus')
  percent_of_same_family = clustering_metrics.percent_same_taxonomy(connected_component, 'family')
  return percent_of_same_genus, percent_of_same_family


def component_string(connected, metadata, metrics):
  output = [output_line(metadata, vlmc) for vlmc in connected]

  metric_string = ("\nPercent of same genus: {:5.5f} \t"
                   "Percent of same family: {:5.5f} \n".format(
                       metrics[0], metrics[1]))

  return '\n'.join(output) + metric_string


def output_line(metadata, vlmc):
  return "{:>55}  {:20} {:20}".format(
      metadata[vlmc.name]['species'],
      metadata[vlmc.name]['genus'],
      metadata[vlmc.name]['family'])
