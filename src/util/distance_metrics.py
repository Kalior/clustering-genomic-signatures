def update_metrics(vlmc, vlmcs, sorted_results, metadata, elapsed_time, metrics):
  procent_genus_in_top = procent_of_taxonomy_in_top(
      vlmc, vlmcs, sorted_results, metadata, 'genus')
  procent_family_in_top = procent_of_taxonomy_in_top(
      vlmc, vlmcs, sorted_results, metadata, 'family')

  average_distance_to_genus = average_distance_to_taxonomy(
      vlmc, sorted_results, metadata, 'genus')
  average_distance_to_family = average_distance_to_taxonomy(
      vlmc, sorted_results, metadata, 'family')
  average_distance = sum([d for d, _ in sorted_results]) / len(sorted_results)

  metrics["procent_genus_in_top"] = procent_genus_in_top
  metrics["procent_family_in_top"] = procent_family_in_top
  metrics["distance_to_genus"] = average_distance_to_genus
  metrics["distance_to_family"] = average_distance_to_family
  metrics["average_distance"] = average_distance

  metrics["average_procent_of_genus_in_top"] += procent_genus_in_top
  metrics["average_procent_of_family_in_top"] += procent_family_in_top
  metrics["total_average_distance_to_genus"] += average_distance_to_genus
  metrics["total_average_distance_to_family"] += average_distance_to_family
  metrics["total_average_distance"] += average_distance
  metrics["global_time"] += elapsed_time

  return metrics


def normalise_metrics(metrics, vlmcs):
  metrics["average_procent_of_genus_in_top"] /= len(vlmcs)
  metrics["average_procent_of_family_in_top"] /= len(vlmcs)
  metrics["total_average_distance"] /= len(vlmcs)
  if metrics["total_average_distance"] == 0:
    metrics["total_average_distance_to_genus"] = 0
    metrics["total_average_distance_to_family"] = 0
  else:
    metrics["total_average_distance_to_genus"] /= (len(vlmcs) * metrics["total_average_distance"])
    metrics["total_average_distance_to_family"] /= (len(vlmcs) * metrics["total_average_distance"])
  metrics["total_average_distance"] = 1

  return metrics


def procent_of_taxonomy_in_top(vlmc, vlmcs, sorted_results, metadata, taxonomy):
  number_same_taxonomy = len([other for other in vlmcs
                              if metadata[other.name][taxonomy] == metadata[vlmc.name][taxonomy]])

  number_same_in_top = 0
  for i in range(number_same_taxonomy):
    _, v = sorted_results[i]
    if metadata[v.name][taxonomy] == metadata[vlmc.name][taxonomy]:
      number_same_in_top += 1

  return number_same_in_top / number_same_taxonomy


def average_distance_to_taxonomy(vlmc, sorted_results, metadata, taxonomy):
  same_taxonomy = [abs(dist) for (dist, other) in sorted_results
                   if metadata[other.name][taxonomy] == metadata[vlmc.name][taxonomy]]
  return sum(same_taxonomy) / len(same_taxonomy)
