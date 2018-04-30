from distance import ACGTContent


def print_distance_output(vlmc, vlmcs, sorted_results, elapsed_time, metadata, metrics):
  print("{}:".format(metadata[vlmc.name]['species']))
  print("Procent of genus in top #genus: {:5.5f} \t"
        "Procent of family in top #family {:5.5f}\n"
        "Average distance to genus: {:5.5f} \t"
        "Average distance to family {:5.5f} \t"
        "Average distance: {:5.5f}".format(
            metrics["procent_genus_in_top"],
            metrics["procent_family_in_top"],
            metrics["distance_to_genus"],
            metrics["distance_to_family"],
            metrics["average_distance"]))

  print("matches self: {}.\tDistance calculated in: {}s\n".format(
      vlmc == sorted_results[0][1], elapsed_time))

  extra_distance = ACGTContent(['C', 'G'])
  result_list = [output_line(metadata, vlmc, dist, v, extra_distance)
                 for (dist, v) in sorted_results]

  print('\n'.join(result_list) + '\n\n')


def output_line(metadata, vlmc, dist, v, d):
  return "{:>55}  {:20} {:20} GC-distance: {:7.5f}   distance: {:10.5f}  {}".format(
      metadata[v.name]['species'],
      metadata[v.name]['genus'],
      metadata[v.name]['family'],
      d.distance(vlmc, v),
      dist,
      same_genus_or_family_string(metadata, vlmc, v))


def same_genus_or_family_string(metadata, vlmc, other_vlmc):
  if metadata[other_vlmc.name]['genus'] == metadata[vlmc.name]['genus']:
    return '  same genus'
  elif metadata[other_vlmc.name]['family'] == metadata[vlmc.name]['family']:
    return '  same family'
  else:
    return ''


def print_metrics(metrics, latex):
  if latex:
    # Name, % genus, % family, dist genus, dist family, time
    print("{:20} & {:10.5f} & {:10.5f} & {:10.5f} & {:10.5f} & {:10.5f}s \\\\ \hline".format(
        metrics['distance_name'],
        metrics['average_procent_of_genus_in_top'],
        metrics['average_procent_of_family_in_top'],
        metrics['total_average_distance_to_genus'],
        metrics['total_average_distance_to_family'],
        metrics['global_time']))
  else:
    print("Distance calculated in: {}s".format(metrics["global_time"]))
    print("Average procent of genus in top #genus: {:5.5f}\t"
          "Average procent of family in top #family {:5.5f}\n"
          "Average distance fraction to genus: {:5.5f}\t"
          "Average distance fraction to family {:5.5f}\t"
          "Average distance: {:5.5f}\n".format(
              metrics["average_procent_of_genus_in_top"],
              metrics["average_procent_of_family_in_top"],
              metrics["total_average_distance_to_genus"],
              metrics["total_average_distance_to_family"],
              metrics["total_average_distance"]))
