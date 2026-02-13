#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(:config no_ignore_case);
use FindBin qw($Bin $Script);

# =============== Variables ===============
my $script_path         = "$Bin";
my $pwd                 = $ENV{"PWD"};
my $user                = $ENV{"USER"};
my $hostname            = $ENV{"HOSTNAME"};
my $library_information = "$Bin/library.txt";
my $docker_image =
  exists $ENV{'DEEPONE_DOCKER_IMAGE'}
  ? $ENV{'DEEPONE_DOCKER_IMAGE'}
  : "sklearn:1.6.1";
my $job_name  = "myjob";
my $partition;
my $host;
my $cpu_num = exists $ENV{'DEEPONE_CPU_NUM'} ? $ENV{'DEEPONE_CPU_NUM'} : 1;
my $gpu_num = exists $ENV{'DEEPONE_GPU_NUM'} ? $ENV{'DEEPONE_GPU_NUM'} : 0;
my $gpu_mode = "gpu";
my $shared_mode;
my $batch_mode;
my $wait_limit;
my $jupyter_mode;
my $debug_mode;
my $docker_cmd;
my $registry = exists $ENV{'DEEPONE_REGISTRY'} ? $ENV{'DEEPONE_REGISTRY'} : undef;
my $sbatch;
my $batch = "slurm.sub";
my $passwd_file = "passwd.temp";
my $group_file = "group.temp";
my $system_cmd;
my @python_versions;
my @docker_images;
my @library_version_list;

# =============== Read library.txt file ===============
my $config = do $library_information;
for my $library_name ( keys %{ $config->{'libraries'} } ) {
    push @docker_images, $library_name;
}

for my $i ( @{ $config->{'python'} } ) {
    push @python_versions, $i;
}

# =============== Get Options ===============
GetOptions(
    'b|batch'       => \$batch_mode,
    'J|job_name=s'  => \$job_name,
    'q|partition=s' => \$partition,
    'm|machine=s'   => \$host,
    'n|cpu=i'       => \$cpu_num,
    'g|gpu=i'       => \$gpu_num,
    's|shared'      => \$shared_mode,
    'w|wait=s'      => \$wait_limit,
    'i|image=s'     => \$docker_image,
    'r|registry=s'  => \$registry,
    'j|jupyter'     => \$jupyter_mode,
    'd|debug'       => \$debug_mode,
    'h|help'        => sub { usage() }
) or usage();

for ( my $i = 0 ; $i < @ARGV ; $i++ ) {

    #========== docker.slurm list ==========
    if ( $ARGV[$i] eq 'list' ) {

        for my $library_name (@docker_images) {
            print "===============\n";
            print "$library_name: ";

            for my $library_version (
                @{ $config->{libraries}{$library_name}{versions} } )
            {
                print " $library_version ";
            }
            print "\n===============\n";
            print $config->{libraries}{$library_name}{description};
            print "\n===============\n";
	    print "\n";
        }
        exit 0;

        #========== docker.slurm exec ==========
    }
    elsif ( $ARGV[$i] eq 'exec' ) {

        my $container = $ARGV[ $i + 1 ]
          or die "ERROR: container name required\n";
        my ($docker_user, $exec_host) = split ' ', `squeue -h -o "%u %N" --job $container`;
	chomp($docker_user) if ($docker_user);
	chomp($exec_host) if ($exec_host);
        if ( $exec_host ne "" && $docker_user eq $user ) {
            if ( $exec_host eq $hostname ) {
                $system_cmd = "docker exec -it $container /bin/bash --login";
            }
            else {
                $system_cmd =
                  "ssh -t $exec_host docker exec -it $container /bin/bash --login";
            }
        }
        else {
            die "$container, Not your container\n";
        }
        print "Executing: $system_cmd\n";
        system($system_cmd) == 0 or die "docker exec failed\n";
        exit 0;

        #========== docker.slurm exec ==========
    }
    elsif ( $ARGV[$i] eq 'kill' ) {
        my $container = $ARGV[ $i + 1 ]
          or die "ERROR: container name required\n";
        my ($docker_user, $exec_host) = split ' ', `squeue -h -o "%u %N" --job $container`;
	chomp($docker_user) if ($docker_user);
	chomp($exec_host) if ($exec_host);
        if ( $exec_host ne "" && $docker_user eq $user ) {
            if ( $exec_host eq $hostname ) {
                $system_cmd = "docker rm -f $container";
            }
            else {
                $system_cmd = "ssh $exec_host docker rm -f $container";
            }
        }
        else {
            die "$container, Not your container\n";
        }
        print "Executing: $system_cmd\n";
        system($system_cmd) == 0 or die "docker exec failed\n";
        exit 0;
    }

}

# =============== Docker command ===============
if ( !-f $library_information || !-r $library_information ) {
    die "$library_information does not exist\n";
}

$docker_cmd =
"docker run --rm -v $pwd:$pwd -w $pwd --name \$SLURM_JOB_ID --hostname d_\$SLURMD_NODENAME ";
# $docker_cmd .= "-v $pwd/$passwd_file:/etc/passwd -v $pwd/$group_file:/etc/group ";
# For none shared file system
$docker_cmd .= "-v /etc/passwd:/etc/passwd -v /etc/group:/etc/group ";
$docker_cmd .= "--gpus $gpu_num "  if ( $gpu_num > 0 );
$docker_cmd .= "-it "              if ( !defined $batch_mode );
$docker_cmd .= "-p \$PORT:\$PORT " if ( defined $jupyter_mode );
$docker_cmd .= "--user \$(id -u):\$(id -g) ";

# Manual port option
# $docker_cmd .= "-p 8888:\$PORT "    if ( defined $jupyter_mode );
if ($registry) {
  $docker_cmd .= "$registry/$docker_image ";
} else {
  $docker_cmd .= "$docker_image ";
}

if ($jupyter_mode) {
    $docker_cmd .= "bash -c \"exec jupyter notebook \\
    --port=\$PORT \\
    --ip=0.0.0.0 \\
    --allow-root \\
    --NotebookApp.token=\"\" \\
    --NotebookApp.custom_display_url=http://\$IP:\$PORT\""
}
elsif (@ARGV) {
    $docker_cmd .= "bash -c \"@ARGV\"";
}
else {
    if ($batch_mode) {
        $docker_cmd .= "tail -f /dev/null";
    }
    else {
        $docker_cmd .= "bash -c \"/bin/bash --login\"";
    }
}

$gpu_mode = "shard" if ($shared_mode);

# =============== BSUB script ===============
$sbatch .= "#!/usr/bin/env bash\n";
$sbatch .= "#SBATCH -J \"$job_name\"\n";
$sbatch .= "#SBATCH -p $partition\n" if ( defined $partition );
$sbatch .= "#SBATCH -w $host\n" if ( defined $host );
$sbatch .= "#SBATCH -n $cpu_num\n";
$sbatch .= "#SBATCH --gres=$gpu_mode:$gpu_num\n" if ( $gpu_num > 0 );
$sbatch .= "#SBATCH -d $wait_limit\n" if ( $wait_limit );

if ($batch_mode) {

    # Batch Job
    $sbatch .= "#SBATCH -o %j.slurmlog\n";
    $sbatch .= "#SBATCH -e %j.slurmlog\n";

    # For Jupyter notebook url
    #  $sbatch .= "#SBATCH -e %j.slurmerror\n";
}
else {
    # Interactive Job
    $system_cmd .= "srun ";
    $system_cmd .= "-J \"$job_name\" ";
    $system_cmd .= "-p $partition " if ( defined $partition );
    $system_cmd .= "-w $host " if ( defined $host );
    $system_cmd .= "-n $cpu_num ";
    $system_cmd .= "--gres=$gpu_mode:$gpu_num " if ( $gpu_num > 0 );
    $system_cmd .= "-d $wait_limit " if ( $wait_limit );

    if ($jupyter_mode) {
        $system_cmd .= "--pty bash -c 'getent passwd > $passwd_file && \\
getent group > $group_file && \\
export IP=\$(hostname -I | awk \"{print \\\$1}\") && \\
export PORT=\$(echo \$SLURM_JOB_ID | tail -c 5) && \\
while [ \${#PORT} -lt 4 ]; do export PORT=\"9\$PORT\"; done && \\
exec $docker_cmd'"; 
    }
    else {
        $system_cmd .= "--pty bash -c 'getent passwd > $passwd_file && \\
getent group > $group_file && \\
exec $docker_cmd'";
    }
}

if ($jupyter_mode) {
    $sbatch .= << 'EOF';

IP=$(hostname -I | awk '{print $1}')
PORT=$(echo $SLURM_JOB_ID | tail -c 5)
while [ ${#PORT} -lt 4 ]; do
    PORT="9$PORT"
done
EOF
}

$sbatch .= "\n";
$sbatch .= "getent passwd > $passwd_file\n";
$sbatch .= "getent group  > $group_file\n";
$sbatch .= $docker_cmd;
$sbatch .= "\n";
$sbatch .= "rm -f $passwd_file\n";
$sbatch .= "rm -f $group_file\n";
$sbatch .= "\n";

open my $SBATCH, ">$batch" or die "Cannot open $batch $!";
print $SBATCH $sbatch;
close $SBATCH;

if ($debug_mode) {
    print "\n";
    print "=====Your Enviroment variables=====\n";
    print "DEEPONE_DOCKER_IMAGE: $ENV{'DEEPONE_DOCKER_IMAGE'}\n"
      if exists $ENV{'DEEPONE_DOCKER_IMAGE'};
    print "DEEPONE_GPU_NUM: $ENV{'DEEPONE_GPU_NUM'}\n"
      if exists $ENV{'DEEPONE_GPU_NUM'};
    print "DEEPONE_REGISTRY: $ENV{'DEEPONE_REGISTRY'}\n"
      if exists $ENV{'DEEPONE_REGISTRY'};
    print "=====Your Enviroment variables=====\n";
    print "\n";
    print "=====Docker command=====\n";
    print $docker_cmd, "\n";
    print "=====Docker command=====\n";
    print "\n";
    print "=====System command=====\n" if ( defined $system_cmd );
    print $system_cmd, "\n" if ( defined $system_cmd );
    print "=====System command=====\n" if ( defined $system_cmd );
    print "\n";
}
elsif ($batch_mode) {
    system("sbatch $batch");
    system("rm -f $batch");
}
else {
    system("rm -f $batch") if ( -e $batch );
    system($system_cmd);
}

sub usage {
    print << "EOF";
=========================================================
docker.slurm 1.0
=========================================================
Usage: docker.slurm [-b] [-d] [-n <cpu_num>] [-g <gpu_num>] [-p <submit_partition>] [-m <submit_host>] -i <docker_image> "commnad"
       docker.slurm list
       docker.slurm exec <JobID>
       docker.slurm kill <JobID>


  -b                 : Batch mode ( Default: Interactive )
  -J <job name>      : Job name  ( Default: myjob)
  -p <partition>     : Sumit partition
  -m <host>          : Sumit host
  -n <cpu num>       : Number of CPU ( Default: 1, DEEPONE_CPU_NUM )
  -g <gpu num>       : Number of GPU ( Default: 0, DEEPONE_GPU_NUM )
  -s                 : Use GPU shared mode
  -w <string>        : Use Job Dependancy
  -j                 : Jupyter notebook mode
  -i <docker_images> : Docker image ( @docker_images ) (DEEPONE_DOCKER_IMAGE)
  -r <registry>      : Docker private registry (DEEPONE_REGISTRY)
  -d                 : Run debug mode

  list          : list docker images
  exec <JobID>  : Go to docker container ( Need ssh ) 
  kill <JobID>  : Kill docker container ( Need ssh ) 
=========================================================
EOF

    exit 0;
}
