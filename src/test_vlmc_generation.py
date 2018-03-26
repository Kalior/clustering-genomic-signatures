from vlmc import VLMC
import parse_trees_to_json
from draw_vlmc import save

import os
import subprocess


def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]


def write_sequence_as_fasta(vlmc_tuple, out_directory, list_path):
  (seq, vlmc) = vlmc_tuple
  file_name = vlmc.name + ".fa"
  print(os.path.join(out_directory, file_name))
  with open(os.path.join(out_directory, file_name), 'w') as f:
    f.write("> {}\n".format(vlmc.name))
    f.write('\n'.join(list(chunks(seq, 990))))

  with open(list_path, 'a') as f:
    f.write(vlmc.name + "\n")


def add_underlines(directory):
  for file in [f for f in os.listdir(directory) if f.endswith(".tree")]:
    name, end = os.path.splitext(file)
    orignal_name = os.path.join(directory, file)
    new_name = os.path.join(directory, name + "__" + end)
    os.rename(orignal_name, new_name)


if __name__ == "__main__":
  out_directory = "../test"
  in_directory = "../original_test_trees"
  image_directory = "../images"
  list_path = os.path.join(out_directory, "list.txt")

  os.system("rm {}/*".format(out_directory))

  parse_trees_to_json.parse_trees(in_directory)
  vlmcs = VLMC.from_json_dir(in_directory)

  sequences = [(vlmc.generate_sequence(25000, 500), vlmc) for vlmc in vlmcs]

  for seq in sequences:
    write_sequence_as_fasta(seq, out_directory, list_path)

  use_constant_cutoff = "false"
  cutoff_value = "3.9075"

  number_of_parameters = 20
  min_count = 20
  max_depth = 20

  standard_args = "-pseudo -crr -f_f {} -ipf .fa -ipwd {} -opwd {} -osf TEST_ -m 1 -frac 0 -revcomp".format(
      list_path, out_directory, out_directory)
  parameter_args = "-c_c {} -nc {} -npar {} -minc {} -kmax {}".format(
      use_constant_cutoff, cutoff_value, number_of_parameters, min_count, max_depth)
  args = ("../lib/classifier " + standard_args + " " + parameter_args).split()
  # ./classifier -pseudo -crr -f_f list.txt -ipf .fa -ipwd test -opwd profiles -osf PST_ -m 1 -frac 0 -revcomp -npar 768 -minc 40 -kmax 9 | tee -a $OUTPUT
  # args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
  popen = subprocess.Popen(args, stdout=subprocess.PIPE)
  popen.wait()
  output = popen.stdout.read()
  print(output)
  # Needed for the parsing (since we expect them to be there...)
  add_underlines(out_directory)
  parse_trees_to_json.parse_trees(out_directory)
  save(in_directory, image_directory)
  save(out_directory, image_directory)
