#!/usr/bin/env perl

# docker command
# docker run --rm -it --name scikit-learn-1.6.0 python:3.12.12 bash -c "echo 'source /python-env/sklearn-1.6.0-env/bin/activate' >> /etc/bash.bashrc && /bin/bash"

use strict;
use warnings;
use Getopt::Long;
use FindBin qw($Bin $Script);

# =============== Variables ===============
my $script_path = "$Bin";
chomp (my $pwd = `pwd`);
my $library_information = "$Bin/library.txt";
my $python_version = exists $ENV{'DEEPONE_PYTHON_VERSION'} ? $ENV{'DEEPONE_PYTHON_VERSION'} : "3.12.12";
my $python_library = exists $ENV{'DEEPONE_PYTHON_LIBRARY'} ? $ENV{'DEEPONE_PYTHON_LIBRARY'} : "sklearn";
my $library_version = exists $ENV{'DEEPONE_LIBRARY_VERSION'} ? $ENV{'DEEPONE_LIBRARY_VERSION'} : "1.8.0";
my $gpu_num = exists $ENV{'DEEPONE_GPU_NUM'} ? $ENV{'DEEPONE_GPU_NUM'} : 0;
my $batch_mode;
my $inputfile;
my $user = $ENV{"USER"};
my $docker_cmd;
my $debug_mode;
my $jupyter_mode;
my @python_versions;
my @python_libraries;
my @library_version_list;

# =============== Read library.txt file ===============
my $config = do $library_information;
for my $library_name (keys %{ $config->{'libraries'} }) {
  push @python_libraries, $library_name;
}

for my $i (@{ $config->{'python'} }) {
  push @python_versions, $i;
}

# =============== Get Options ===============
GetOptions(
  'b|batch'     => \$batch_mode,
  'i|input=s' => \$inputfile,
  'v|ver=s' => \$python_version,
  'l|lib=s' => \$python_library,
  'lv|library_version=s' => \$library_version,
  'g|gpu=i' => \$gpu_num,
  'j|jupyter' => \$jupyter_mode,
  'd|debug' => \$debug_mode,
  'h|help' => sub{ usage() }
) or usage();

for my $arg (@ARGV) {
  # command) ai_sub list 
  if ( $arg =~ /\blist\b/ ) {
    # print all python libraries and verion
    for my $library_name (@python_libraries) {
      print "===============\n";
      print $config->{'libraries'}->{"$library_name"}->{'description'}, "\n";
      print "===============\n";
      print "$library_name: ";
      for my $library_version (@{ $config->{'libraries'}->{"$library_name"}->{'versions'} }) {
        print " $library_version ";
      }
      print "\n===============\n\n";
    }
    exit 0;
  }
}


# =============== Main logic ===============
if ( ! -f $library_information || ! -r $library_information ) {
  die "$library_information does not exist\n";
}

$docker_cmd = "docker run --rm \\";
$docker_cmd .= "--gpus $gpu_num " if ( $gpu_num > 0 );
if ( ! defined $batch_mode && ! defined $jupyter_mode ) {
  $docker_cmd .= "-it \\";
}
if ( defined $jupyter_mode ) {
$docker_cmd .= "-p 8888:8888 \\";
}
$docker_cmd .= "
-v $pwd:$pwd \\
-w $pwd \\
--name \"${library_version}-${user}\" \\
python:$python_version \\
";

if ( $batch_mode ) {
  die "Need -i inputfile." if ( ! defined $inputfile );
  $docker_cmd .= "bash -c \"source /python-env/${python_library}-${library_version}-env/bin/activate && python3 $inputfile\"";
} elsif ( $jupyter_mode ){
  $docker_cmd .= "bash -c \"source /python-env/${python_library}-${library_version}-env/bin/activate && jupyter notebook \\
    --port=8888 \\
    --ip=0.0.0.0 \\
    --allow-root \\
    --NotebookApp.token='' \\
    --NotebookApp.custom_display_url=http://127.0.0.1:8888\""
}
  else {
  $docker_cmd .= "bash -c \"echo 'source /python-env/${python_library}-${library_version}-env/bin/activate' >> /etc/bash.bashrc && /bin/bash\"";
}

if ( $debug_mode ) {
  # ./ai_sub -d
  print "\n";
  print "=====Your Enviroment variables=====\n";
  print "DEEPONE_PYTHON_VERSION: $ENV{'DEEPONE_PYTHON_VERSION'}\n" if exists $ENV{'DEEPONE_PYTHON_VERSION'};
  print "DEEPONE_PYTHON_LIBRARY: $ENV{'DEEPONE_PYTHON_LIBRARY'}\n" if exists $ENV{'DEEPONE_PYTHON_LIBRARY'};
  print "DEEPONE_LIBRARY_VERSION: $ENV{'DEEPONE_LIBRARY_VERSION'}\n" if exists $ENV{'DEEPONE_LIBRARY_VERSION'};
  print "DEEPONE_GPU_NUM: $ENV{'DEEPONE_GPU_NUM'}\n" if exists $ENV{'DEEPONE_GPU_NUM'};
  print "=====Your Enviroment variables=====\n";
  print "\n";
  print "=====Docker command=====\n";
  print $docker_cmd, "\n";
  print "=====Docker command=====\n";
  print "\n";
  exit 0;
} else {
  # run docker command
  system($docker_cmd);
}

sub usage {
  print << "EOF";
=========================================================
ai_sub 1.0.0 (venv base)
=========================================================
Usage: $0 -g <gpu_num> -v <python version> -l <python library> -lv <python library version>
       $0 list

  -b  |  --batch             : Batch mode ( Default Interactive )
  -i  |  --input             : Input file
  -g  |  --gpu               : Number of gpu ( DEEPONE_GPU_NUM )
  -v  |  --version           : Python version ( @python_versions ) (DEEPONE_PYTHON_VERSION)
  -l  |  --lib               : Python library ( @python_libraries ) (DEEPONE_PYTHON_LIBRARY)
  -lv |  --library_version   : Python library version (DEEPONE_LIBRARY_VERSION)

  Optional command)

  list                       : list python libraries and version
=========================================================
EOF

exit 0
}
