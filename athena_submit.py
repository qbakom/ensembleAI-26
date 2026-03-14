"""
Transfer solutions to Athena and submit via SLURM.
Uses paramiko for SSH/SFTP.
"""
import paramiko, os, sys, time

HOST = "athena.cyfronet.pl"
USER = "tutorial243"
PASS = "pybtujumra"
TEAM_TOKEN = "b2c6083ba78b4039a6db64a4bb5e07ca"
SERVER_URL = "http://149.156.182.9:6060"

LOCAL_PACKAGE = "/srv/root/ensembleAI-26/athena_package"

def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=30)
    return ssh

def run_cmd(ssh, cmd, timeout=120):
    print(f"  CMD: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out.strip():
        print(f"  OUT: {out.strip()}")
    if err.strip():
        print(f"  ERR: {err.strip()}")
    return out, err

def upload_files(ssh):
    sftp = ssh.open_sftp()
    
    # Get SCRATCH path
    out, _ = run_cmd(ssh, 'echo $SCRATCH')
    scratch = out.strip()
    remote_dir = f"{scratch}/hackathon"
    
    # Create remote dirs
    for subdir in ['', '/task1', '/task2', '/task3', '/task4']:
        try:
            sftp.mkdir(f"{remote_dir}{subdir}")
        except:
            pass
    
    # Upload all files in package
    for root, dirs, files in os.walk(LOCAL_PACKAGE):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, LOCAL_PACKAGE)
            remote_path = f"{remote_dir}/{rel_path}"
            print(f"  Upload: {rel_path} -> {remote_path}")
            sftp.put(local_path, remote_path)
    
    sftp.close()
    return remote_dir

def submit_interactive(ssh, remote_dir):
    """Submit using srun interactive job."""
    submit_script = f"""
export TEAM_TOKEN="{TEAM_TOKEN}"
export SERVER_URL="{SERVER_URL}"
cd {remote_dir}

module add GCCcore/13.2.0 Python/3.11.5

VENV_DIR=$SCRATCH/venvs/hackathon
if [ ! -d "$VENV_DIR" ]; then
    mkdir -p $SCRATCH/venvs
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip --no-cache-dir --require-virtualenv install requests 2>&1 | tail -1

echo "=== Task 1 ==="
[ -f task1/chebi_submission.parquet ] && python3 submit_solution.py task1 task1/chebi_submission.parquet parquet_file || echo "SKIP task1"

echo "=== Task 2 (public) ==="  
[ -f task2/python-public-smart.jsonl ] && python3 submit_solution.py task2 task2/python-public-smart.jsonl jsonl_file public || echo "SKIP task2"

echo "=== Task 3 ==="
[ -f task3/load_submission.csv ] && python3 submit_solution.py task3 task3/load_submission.csv csv_file || echo "SKIP task3"

echo "=== Task 4 ==="
[ -f task4/ecg_submission.npz ] && python3 submit_solution.py task4 task4/ecg_submission.npz npz_file || echo "SKIP task4"

echo "=== DONE ==="
"""
    # Write script to remote
    sftp = ssh.open_sftp()
    script_path = f"{remote_dir}/run_submit.sh"
    with sftp.open(script_path, 'w') as f:
        f.write(submit_script)
    sftp.close()
    
    # Submit via sbatch
    slurm_script = f"""#!/bin/bash -l
#SBATCH -J hackathon-sub
#SBATCH -N 1
#SBATCH --ntasks-per-node 2
#SBATCH --time=00:20:00
#SBATCH -A tutorial
#SBATCH -p tutorial
#SBATCH --mem=4GB
#SBATCH --output={remote_dir}/submit_%J.out

bash {remote_dir}/run_submit.sh
"""
    sftp = ssh.open_sftp()
    slurm_path = f"{remote_dir}/submit.slurm"
    with sftp.open(slurm_path, 'w') as f:
        f.write(slurm_script)
    sftp.close()
    
    out, err = run_cmd(ssh, f"sbatch {slurm_path}")
    return out

def check_status(ssh):
    out, _ = run_cmd(ssh, "squeue --me")
    return out

def main():
    print("Connecting to Athena...")
    ssh = connect()
    
    print("\nUploading files...")
    remote_dir = upload_files(ssh)
    
    print(f"\nFiles uploaded to: {remote_dir}")
    
    print("\nSubmitting SLURM job...")
    result = submit_interactive(ssh, remote_dir)
    
    print("\nChecking queue...")
    check_status(ssh)
    
    # Wait for job to complete
    print("\nWaiting for job to complete (checking every 30s)...")
    for i in range(20):
        time.sleep(30)
        out, _ = run_cmd(ssh, "squeue --me --noheader | wc -l")
        n_jobs = int(out.strip())
        if n_jobs == 0:
            print("Job completed!")
            break
        print(f"  Still running ({n_jobs} jobs)...")
    
    # Show output
    print("\n=== Submission Output ===")
    out, _ = run_cmd(ssh, f"cat {remote_dir}/submit_*.out 2>/dev/null")
    
    ssh.close()

if __name__ == "__main__":
    main()
